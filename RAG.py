from lightrag import LightRAG, QueryParam
from lightrag.utils import setup_logger
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
import kconfiglib as klib

import asyncio
import os
from neo4j import GraphDatabase


class KnowledgeGenerator:
    def __init__(
        self,
        query,
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
        setup_logger("lightrag", "ERROR")
        asyncio.run(self.__async_init())
        # self.gen_configs_knowledge()
        self.search_mode = search_mode
        self.gk = gen_knowledge
        if not self.gk:
            print("RAG will not generate knowledge")
        
        # init neo4j
        uri = os.environ['NEO4J_URI']
        username = os.environ['NEO4J_USERNAME']
        password = os.environ['NEO4J_PASSWORD']
        self.driver = GraphDatabase.driver(uri=uri, auth=(username, password))
        
        self.tags = self.query_to_tag(query)

    async def __async_init(self, working_dir, model_func):
        self.rag = LightRAG(
            working_dir=working_dir,
            embedding_func=openai_embed,
            llm_model_func=model_func,
            graph_storage="Neo4JStorage"
        )
        await self.rag.initialize_storages()
        await initialize_pipeline_status()
    
    def query_to_tag(self, target: str) -> set[str]:
        # add query into rag
        self.rag.insert(target)
        # connect
        query = f"Given target = '{target}', which entities might related to this target? Answer the entity name only."
        results = self.rag.query(query)
        result_list = results.split('\n')
        tags = [result[2:] for result in result_list]
        return tags

    def gen_knowledge(self, prompt: str):
        return self.rag.query(prompt, param=QueryParam("hybrid"))

    def gen_configs_knowledge(self, configs: list[klib.MenuNode], target: str):
        if not self.gk:
            return ""
        prompt = f"Of these configs listed below, which ones may affect the target?: {target}\n"

        def get_simple_node_name(node: klib.MenuNode):
            item = node.item
            if hasattr(item, "name"):
                return item.name
            else:
                return node.prompt[0]
        config_name_list = []

        for config in configs:
            item = config.item
            if item == klib.MENU:
                prompt += config.prompt[0]
                prompt += "\n"
                config_name_list.append(get_simple_node_name(config))
            elif isinstance(item, klib.Choice):
                prompt += config.prompt[0]
                prompt += "\n"
                config_name_list.append(get_simple_node_name(config))
            elif isinstance(item, klib.Symbol):
                prompt += item.name
                prompt += "\n"
                config_name_list.append(get_simple_node_name(config))
        knowledge = self.gen_knowledge(prompt)

        # append additional knowledge
        def get_tags_by_config(self, config: str):
            query = """
MATCH (n)-[:HAS_TAG]->(m)
WHERE m.name = $config_name
RETURN n.entity_id
"""
            records = self.driver.execute_query(query, config_name=config).records
            s: set = ()
            for record in records:
                s.add(record.value())
            return s
        additional_knowledge = f"""\nAdditionally, these information might also be useful:
TARGET {target} might related to these tags: {', '.join(self.tags)}. The relationship of these configs and these tags are listed below:\n
"""
        for config_name in config_name_list:
            tags = get_tags_by_config(config_name)
            union = tags & self.tags
            if len(union) > 0:
                additional_knowledge += f"config '{config_name}' might affect tags {union}\n"

        return knowledge + additional_knowledge

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
    
    def add_tag(self, tags, configs):
        query = """MATCH (n {entity_id: "$tag"}), (m {entity_id: "$config"})
CREATE (n)-[:HAS_TAG]->(m)"""
        for tag in tags:
            for config in configs:
                self.driver.execute_query(query, tag = tag, config=config)
    def delete_tag(self, tags, configs):
        query = """MATCH (n {entity_id: "$tag"})-[r:HAS_TAG]->(m {entity_id: "$config"})
DELETE r"""
        for tag in tags:
            for config in configs:
                self.driver.execute_query(query, tag=tag, query=query)
