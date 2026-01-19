import kconfiglib as klib
import logging
from RAG import KnowledgeGenerator
import os
import json


class Config:
    """Configuration builder that traverses a Kconfig tree and uses an LLM
    (optionally with RAG) to decide which options to enable/disable and what
    values to set.

    The class orchestrates:
    - parsing Kconfig via kconfiglib
    - asking an LLM (through `chatter`) to select menus/options
    - optionally enriching prompts with knowledge from `KnowledgeGenerator`
    - logging QA pairs to `QA.log` for future training
    """

    def __init__(
        self,
        kconfig_path: str,
        chatter,
        target: str,
        kg_search_mode: str,
        use_knowledge: bool,
        config_path: str = ".config",
        debug: bool = False,
    ):
        # Parse the Kconfig file and load existing .config values (if present).
        self.kconfig = klib.Kconfig(kconfig_path)
        self.kconfig.load_config(config_path)

        # LLM interface used to query human-like decisions about config items
        self.chatter = chatter

        # traversal state: current menu node and stack of unvisited menu nodes
        self.current_node: klib.MenuNode = self.kconfig.top_node
        self.unvisit_node_list: list[klib.MenuNode] = [self.kconfig.top_node]

        # mapping from MenuNode -> list of strings representing the menu path
        # used for printing and logging hierarchical menu names
        self.node_dir_dict: dict[klib.MenuNode, list[str]] = {
            self.kconfig.top_node: [self.kconfig.top_node.prompt[0]]
        }

        # configure file logging for config decisions
        logging.basicConfig(
            level=logging.INFO,
            filename="Config.log",
            datefmt="%Y/%m/%d %H:%M:%S",
        )

        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.FileHandler("Config.log", mode="w"))
        self.logger.propagate = False
        self.logger.info(target)

        # question logger: store QA examples (question + chosen answers) to QA.log
        # This file can be used later as supervised training data.
        self.qlogger = logging.getLogger("__question_logger__")
        self.qlogger.addHandler(logging.FileHandler("QA.log", mode="w"))
        self.qlogger.propagate = False
        self.qlogger.info(target)

        self.target = target

        # RAG knowledge generator that can enrich LLM prompts. If gen_knowledge is
        # False, it should produce empty or minimal context.
        self.kg = KnowledgeGenerator(
            working_dir=os.environ['WORKING_DIR'],
            search_mode=kg_search_mode,
            gen_knowledge=use_knowledge,
        )

        self.debug = debug

    def run(self):
        """Main loop: visit menu nodes until none remain.

        The traversal is LIFO: nodes are popped from `unvisit_node_list`.
        For each visited menu we call `process()` to decide on its children.
        """
        while len(self.unvisit_node_list) > 0:
            self.current_node = self.unvisit_node_list.pop()
            print(f"Visiting menu {'/'.join(self.node_dir_dict[self.current_node])}")
            self.process()
    
    def process(self):
        """Classify children of the current node and process each category.

        Node categories:
        - menu_nodes: sub-menus to potentially expand
        - bool_nodes: on/off options (may enable menus)
        - binary_nodes/trinary_nodes: tristate options (not implemented)
        - multiple_nodes: choice/select options
        - value_nodes: string/int/hex settings
        After classification we call the appropriate handlers. Menu nodes are
        not automatically expanded — `extend_nodes` asks the LLM which menus to
        actually traverse next.
        """
        menu_nodes = []
        bool_nodes = []
        binary_nodes = []
        trinary_nodes = []
        multiple_nodes = []
        value_nodes = []

        # collect the active child nodes of the current menu
        nodes = self.get_menunodes(self.current_node)

        # categorize nodes by type so different handlers can process them
        for node in nodes:
            item = node.item  # determine node type through this property
            if item == klib.MENU:
                menu_nodes.append(node)
            elif item == klib.COMMENT:
                # comments are ignored for now
                pass
            else:
                # value nodes accept explicit values (string/int/hex)
                if item.type in (klib.STRING, klib.INT, klib.HEX):
                    value_nodes.append(node)
                # visible choice nodes (select / choice) treated as multiple-choice
                elif (
                    isinstance(item, klib.Choice)
                    and item.visibility == 2
                    and item.str_value == "y"
                ):
                    multiple_nodes.append(node)
                # nodes that are effectively always 'y' but still contain a child list
                elif len(item.assignable) == 1 and node.list:
                    menu_nodes.append(node)
                # boolean nodes (on/off)
                elif item.type == klib.BOOL:
                    bool_nodes.append(node)
                # tristate nodes (n/m/y) - separate binary-vs-trinary handling
                elif item.type == klib.TRISTATE:
                    if item.assignable == (1, 2):
                        binary_nodes.append(node)
                    else:
                        trinary_nodes.append(node)

        # process each category using dedicated handlers
        if len(multiple_nodes) != 0:
            self.process_multiple(multiple_nodes)
        if len(value_nodes) != 0:
            self.process_value(value_nodes)
        if len(binary_nodes) != 0:
            self.process_binary(binary_nodes)
        if len(trinary_nodes) != 0:
            self.process_trinary(trinary_nodes)

        # boolean options might reveal sub-menus when enabled; collect them
        new_menu_nodes = []
        if len(bool_nodes) != 0:
            new_menu_nodes.extend(self.process_bool(bool_nodes))

        # extend the menu list with any new menus revealed by enables
        menu_nodes.extend(new_menu_nodes)
        if len(menu_nodes) != 0:
            # ask the LLM which of the candidate menu nodes should be expanded
            self.unvisit_node_list.extend(self.extend_nodes(menu_nodes))

    def get_menunodes(self, node: klib.MenuNode) -> list[klib.MenuNode]:
        """Return active child nodes of a menu node.

        The method walks the linked-list of children (`node.list` / `.next`) and
        includes nodes whose prompt condition evaluates to true according to
        `klib.expr_value` and whose type is supported.
        """
        node: klib.MenuNode = node.list
        # collect nodes that are visible/active under the current menu
        node_list = []
        while node:
            if node.prompt:  # if node.prompt exists (title/help prompt)
                if klib.expr_value(node.prompt[1]):
                    item = node.item
                    # include symbol/choice nodes unless they are UNKNOWN type
                    if isinstance(item, klib.Symbol) or isinstance(item, klib.Choice):
                        if item.type != klib.UNKNOWN:
                            node_list.append(node)
                    else:
                        # include menus and other node types
                        node_list.append(node)
            node = node.next
        return node_list

    def extend_nodes(self, nodes: list[klib.MenuNode]) -> list[klib.MenuNode]:
        """Ask the LLM which menu nodes (sub-menus) should be expanded next.

        To keep prompts small we batch nodes into groups of 3 and generate
        RAG-based knowledge for each batch. The LLM is expected to return a
        list of selections, either indices (int) or exact node name strings.
        """
        # produce supporting knowledge for node groups (3 per group)
        knowledge_group = []
        for i in range(0, len(nodes), 3):
            knowledge_group.append(nodes[i : i + 3])
        knowledges = [
            self.kg.gen_configs_knowledge(g, self.target) for g in knowledge_group
        ]
        knowledge = "\n".join(knowledges)

        # create readable lines like "0 Menu name (SYMBOL)" for the LLM
        node_name_list = []
        node_name_list_without_idx = []
        node_name_dict = {}
        for i in range(len(nodes)):
            node_name = self.get_node_name(nodes[i])
            node_name_list.append(f"{i} {node_name}")
            node_name_list_without_idx.append(node_name)
            node_name_dict[node_name] = nodes[i]

        # query the LLM with the numbered menu list and optional knowledge
        content = "\n".join(node_name_list)
        answers = self.chatter.ask_menu(content=content, knowledge=knowledge)
        # expected answers: list of ints (indices) and/or strings (node names)

        menu_node: list[klib.MenuNode] = []
        # prefix path for nodes selected under the current menu
        path = self.node_dir_dict[self.current_node]

        # collect entries for QA logging
        qlogger_ans = []
        for answer in answers:
            if type(answer) == int:
                try:
                    node = nodes[answer]
                    menu_node.append(node)
                    # record hierarchical path for the selected node
                    self.node_dir_dict[node] = path + [node.prompt[0]]
                    qlogger_ans.append(node.prompt[0])
                except IndexError:
                    if self.debug:
                        print(f"LLM gives non-existent nodes(int). current node is\n{nodes}\nLLM gives\n{answer}")
            else:
                # string answers must match exactly one of the node_name keys
                if answer in node_name_dict.keys():
                    if answer.isspace() or answer == "" or answer == "\n":
                        continue
                    node = node_name_dict[answer]
                    menu_node.append(node)
                    self.node_dir_dict[node] = path + [node.prompt[0]]
                    qlogger_ans.append(node.prompt[0])
                else:
                    if self.debug:
                        print(f"LLM gives non-existent nodes(string). current node is\n{nodes}\nLLM gives\n{answer}")

        # persist the QA pair (presented menu and chosen subset) for training data
        if len(qlogger_ans) > 0:
            self.qlogger.info(
                json.dumps({"question": "Menu\t" + "\n".join(node_name_list_without_idx), "answer": qlogger_ans})
            )
        return menu_node

    def process_bool(self, nodes: list[klib.MenuNode]) -> list[klib.MenuNode]:
        """Handle boolean options by asking the LLM to turn them on/off.

        The function batches up to 9 boolean options per LLM call. The LLM is
        expected to return a mapping from config name -> integer state
        (0/2 where 0=off, 2=on). Any options enabled that reveal a submenu are
        returned so callers can later decide whether to expand those menus.
        """
        new_menu_nodes_dict: dict[str, klib.MenuNode] = {}
        # batch into groups of up to 9 to limit LLM prompt size
        nodes_group = []
        for i in range(0, len(nodes), 9):
            nodes_group.append(nodes[i : i + 9])

        for group in nodes_group:
            node_name_dict = {}
            node_name_lower_dict = {}
            for node in group:
                name = self.get_node_name(node)
                node_name_dict[name] = node
                simple_name = self.get_simple_node_name(node)
                node_name_lower_dict[simple_name.lower()] = node
                # if already enabled and has children, prepare to expand it
                if node.item.tri_value == 2 and node.list:
                    new_menu_nodes_dict[name.lower()] = node
            node_names = "\n".join(node_name_dict.keys())

            # generate RAG knowledge in batches of 3 within this group
            knowledge_group = []
            for i in range(0, len(group), 3):
                knowledge_group.append(group[i : i + 3])
            knowledges = [
                self.kg.gen_configs_knowledge(g, self.target) for g in knowledge_group
            ]
            knowledge = "\n".join(knowledges)

            # the chatter should return a dict mapping config name -> selected state
            answer = self.chatter.ask_on_off_option(node_names, knowledge)

            # record answers for qlogger and apply changes
            qlogger_ans = []
            for config_name, state in answer.items():
                config_name = config_name.strip().lower()
                if config_name in node_name_lower_dict.keys():
                    node = node_name_lower_dict[config_name]
                    qlogger_ans.append({"config": config_name, "value": state})
                    if node.item.tri_value == state:
                        # no change needed
                        continue
                    # log previous/changed state and set the new value
                    self.logger.info(
                        f"CONFIG_{node.item.name}={node.item.str_value}"
                    )
                    # set config value (0=off, 2=on for booleans/tristates)
                    node.item.set_value(state)
                    # if enabling reveals a submenu, add to expansion candidates
                    if state == 2:
                        new_menu_nodes_dict[config_name] = node
                    elif state == 0 and config_name in new_menu_nodes_dict.keys():
                        new_menu_nodes_dict.pop(config_name)
                else:
                    if self.debug:
                        print(f"Error: config name {config_name} does not exist")
                        print(f"All configs: {node_name_lower_dict.keys()}")
            # persist QA examples for this group
            if len(qlogger_ans) > 0:
                self.qlogger.info(json.dumps({"question": "Bool\t" + node_names, "answer": qlogger_ans}))
        return new_menu_nodes_dict.values()

    def process_binary(self, nodes: list[klib.MenuNode]):
        # Binary/tristate handlers are not implemented yet. They should mirror
        # `process_bool` but handle the allowed values specific to tristate/binary
        # symbols and possibly multiple-choice/ordering rules.
        pass

    def process_trinary(self, nodes: list[klib.MenuNode]):
        # Placeholder for handling 3-state options where all three values may be
        # selectable (e.g., builtin/module/disabled). Implementation left as an
        # exercise if needed for the target Kconfig set.
        pass

    def process_multiple(self, nodes: list[klib.MenuNode]):
        """Handle choice/select nodes where one option should be chosen.

        For each choice node we present the available alternatives and ask the
        LLM to select one. The chosen option is set to 'y' (tri_value == 2).
        """
        for node in nodes:
            choices: list[klib.MenuNode] = []
            node_list = self.get_menunodes(node)
            for choice in node_list:
                choices.append(choice)

            # query LLM with choice list and optional RAG knowledge for those choices
            answer = self.chatter.ask_multiple_option(
                "\n".join([self.get_node_name(choice) for choice in choices]),
                self.kg.gen_configs_knowledge(choices, self.target),
            ).strip()

            # tolerate bracketed list output like "[option]"
            if answer.startswith("[") and answer.endswith("]"):
                answer = answer[1:-1]

            # find selected answer by comparing to simple node name
            found = False
            qlogger_ans = None
            for choice in node_list:
                if answer == self.get_simple_node_name(choice):
                    qlogger_ans = answer
                    # mark the selected choice as active if not already
                    if choice.item.tri_value != 2:
                        self.logger.info(f"CONFIG_{choice.item.name}=y")
                        choice.item.set_value(2)
                    found = True
            if not found:
                if self.debug:
                    print(f"Error: answer {answer} does not exist")
                    configs = "\n".join([self.get_simple_node_name(choice) for choice in choices])
                    print(f"All configs: {configs}")
            else:
                # log QA pair for training dataset
                self.qlogger.info(
                    json.dumps({
                        "question": "Choice\t" + "\n".join([self.get_node_name(choice) for choice in choices]),
                        "answer": qlogger_ans,
                    })
                )

    def process_value(self, nodes: list[klib.MenuNode]):
        """Handle nodes that accept explicit values (string/int/hex).

        We construct a human-readable prompt for each symbol and ask the LLM
        to provide values. The LLM wrapper must return a list of tuples where
        each tuple is (prompt_text, chosen_value).
        """
        # build prompt lines and mapping from prompt -> node for postprocessing
        help_info_list = []
        node_info_list = []
        prompt_to_node_dict = {}

        # helper to create a help prompt similar to menuconfig's presentation
        def get_help_info_from_sym(sym: klib.Symbol):
            tristate_name = ["n", "m", "y"]
            prompt = f"Value for {sym.name}"
            if sym.type in (klib.BOOL, klib.TRISTATE):
                prompt += f" (available: {', '.join(tristate_name[val] for val in sym.assignable)})"
            prompt += ":"
            return f"{str(sym)}\n{prompt}"

        for node in nodes:
            item = node.item
            help_info_list.append(get_help_info_from_sym(item))
            node_info_list.append(f"{node.prompt[0]} ({item.str_value})")
            # map the display prompt (node.prompt[0]) to its MenuNode for later
            prompt_to_node_dict[node.prompt[0]] = node

        # query LLM for values; the chatter should parse and return structured answers
        answers = self.chatter.ask_value_option("\n".join(help_info_list), "\n".join(node_info_list))

        # postprocess returned tuples and set symbol values accordingly
        qlogger_ans = []
        # answers is expected to be a list of tuples: (prompt_text, chosen_value)
        for answer in answers:
            if answer[0] in prompt_to_node_dict.keys():
                node = prompt_to_node_dict[answer[0]]
                qlogger_ans.append({"config": answer[0], "value": answer[1]})
                if node.item.str_value == answer[1]:
                    continue
                # log and set the new value
                self.logger.info(f"CONFIG_{node.item.name}={answer[1]}")
                prompt_to_node_dict[answer[0]].item.set_value(answer[1])

        # persist QA examples if any values were changed or suggested
        if len(qlogger_ans) > 0:
            self.qlogger.info(json.dumps({"question": "Value\t" + "\n".join(node_info_list), "answer": qlogger_ans}))

    def save(self, path: str):
        self.kconfig.write_config(path)
        os.rename("Config.log", path + ".log")
        l = path.split('/')
        l[-1] = "QA_" + l[-1]
        os.rename("QA.log", '/'.join(l) + ".log")

    def get_node_name(self, node: klib.MenuNode):
        name = node.prompt[0]
        item = node.item
        if hasattr(item, "name"):
            name = f"{name} ({item.name})"
        return name

    def get_simple_node_name(self, node: klib.MenuNode):
        item = node.item
        if hasattr(item, "name"):
            return item.name
        else:
            return node.prompt[0]
