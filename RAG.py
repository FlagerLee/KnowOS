from lightrag import LightRAG, QueryParam
from lightrag.utils import setup_logger
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
import kconfiglib as klib


class KnowledgeGenerator:
    def __init__(
        self,
        working_dir: str,
        gen_knowledge: bool,
        search_mode: str,
        llm_model_func: str = "gpt-4o-mini",
    ):
        model_func_map = {"gpt-4o-mini": gpt_4o_mini_complete, "gpt-4o": gpt_4o_complete}
        model_func = (
            model_func_map[llm_model_func]
            if llm_model_func in model_func_map.keys()
            else gpt_4o_mini_complete
        )
        self.rag = LightRAG(
            working_dir=working_dir, llm_model_func=model_func, embedding_func=openai_embed
        )
        setup_logger("lightrag", "ERROR")
        self.search_mode = search_mode
        self.gk = gen_knowledge
        if not self.gk:
            print("RAG will not generate knowledge")

    def gen_knowledge(self, prompt: str):
        return self.rag.query(prompt, param=QueryParam("hybrid"))

    def gen_configs_knowledge(self, configs: list[klib.MenuNode], target: str):
        if not self.gk:
            return ""
        prompt = f"Of these configs listed below, which ones may affect the target?: {target}\n"
        for config in configs:
            item = config.item
            if item == klib.MENU:
                prompt += config.prompt[0]
                prompt += "\n"
            elif isinstance(item, klib.Choice):
                prompt += config.prompt[0]
                prompt += "\n"
            elif isinstance(item, klib.Symbol):
                prompt += item.name
                prompt += "\n"
        return self.gen_knowledge(prompt)

    def init_config_storage(self, config):
        def init_config(node: klib.MenuNode) -> tuple[list[dict], list[dict], str]:
            entities = []
            item = node.item
            name, h = "", ""
            if item == klib.MENU:
                name = node.prompt[0]
                h = ""
                if hasattr(node, "help"):
                    h = node.help
            elif item == klib.COMMENT:
                return [], [], None
            elif isinstance(item, klib.Symbol) or isinstance(item, klib.Choice):
                if node.item.name:
                    name = node.item.name
                    h += node.prompt[0]
                    h += "\n"
                    if hasattr(node, "help") and node.help:
                        h += node.help
                else:
                    name = node.prompt[0]
                    if hasattr(node, "help") and node.help:
                        h += node.help
            entities.append({
                "entity_name": name,
                "entity_type": "config",
                "description": h,
                "source_id": "Kconfig"
            })
            relationships = []
            child = node.list
            while child:
                if not child.prompt:
                    child = child.next
                    continue
                e, r, n = init_config(child)
                if not n:
                    child = child.next
                    continue
                entities.extend(e)
                relationships.extend(r)
                relationships.append({
                    "src_id": name,
                    "tgt_id": n,
                    "description": f"{name} is parent of {n}",
                    "keywords": "config",
                    "weight": 1.0,
                    "source_id": "Kconfig"
                })
                child = child.next
            return entities, relationships, name
        e, r, n = init_config(config)
        self.rag.insert_custom_kg({
            "chunks": [],
            "entities": e,
            "relationships": r
        })
