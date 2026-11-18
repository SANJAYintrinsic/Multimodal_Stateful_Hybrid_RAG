"""
Knowledge Graph module: extracts entities/relationships from text chunks
and stores them in a NetworkX graph. The graph is saved to disk (pickle file)
so it survives app restarts, not just in-memory during one session.
"""

import os
import pickle

import spacy
import networkx as nx

from src.config import SPACY_MODEL, KG_STORE_PATH


class KnowledgeGraphBuilder:
    def __init__(self):
        self.nlp = spacy.load(SPACY_MODEL)
        self._add_custom_entity_ruler()
        self.graph = nx.Graph()
        self._load()

    def _add_custom_entity_ruler(self):
        """
        spaCy's statistical NER relies heavily on capitalization, so short
        acronyms/company names typed in lowercase (common in casual test
        text or OCR output) often get missed. This adds a rule-based layer
        that recognizes a curated list of common organizations regardless
        of case, running alongside (not replacing) the normal NER model.
        """
        known_orgs = [
            "tcs", "infosys", "wipro", "hcl", "ibm", "google", "microsoft",
            "amazon", "cognizant", "accenture", "capgemini", "tech mahindra",
            "deloitte", "meta", "apple", "oracle", "sap", "adobe", "intel",
        ]
        patterns = [{"label": "ORG", "pattern": [{"LOWER": org}]} for org in known_orgs]

        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(patterns)

    # ---------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------
    def _load(self):
        if os.path.exists(KG_STORE_PATH):
            with open(KG_STORE_PATH, "rb") as f:
                self.graph = pickle.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(KG_STORE_PATH), exist_ok=True)
        with open(KG_STORE_PATH, "wb") as f:
            pickle.dump(self.graph, f)

    def clear(self):
        self.graph.clear()
        self._save()

    # ---------------------------------------------------------------
    # Building the graph
    # ---------------------------------------------------------------
    
    def add_text(self, text: str, source: str):
        """
        Links entities that co-occur within the SAME PARAGRAPH, rather than
        a fixed sentence window. This fits short (<=250 word), single-topic
        documents well: a paragraph is naturally one coherent idea, so
        entities within it - even several sentences apart, or connected only
        by a pronoun - are fairly assumed to be related.

        Paragraphs are split on blank lines (double newlines). A document
        with no blank lines at all is treated as one single paragraph.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else []

        for para in paragraphs:
            doc = self.nlp(para)
            entities = []
            for ent in doc.ents:
                ent_text = ent.text.strip()
                if ent_text and ent_text not in entities:
                    entities.append(ent_text)

            for ent in entities:
                if not self.graph.has_node(ent):
                    self.graph.add_node(ent, sources={source})
                else:
                    self.graph.nodes[ent]["sources"].add(source)

            # Use a shortened version of the paragraph as the "evidence" text
            # shown in KG facts, so it stays readable rather than dumping the
            # whole paragraph back into the chat.
            evidence = para if len(para) <= 200 else para[:200].rsplit(" ", 1)[0] + "..."

            for a in range(len(entities)):
                for b in range(a + 1, len(entities)):
                    x, y = entities[a], entities[b]
                    if self.graph.has_edge(x, y):
                        self.graph[x][y]["weight"] += 1
                    else:
                        self.graph.add_edge(x, y, weight=1, sentence=evidence)

        self._save()

    # ---------------------------------------------------------------
    # Querying
    # ---------------------------------------------------------------
    
    def related_entities(self, query: str):
        """
        Find entities relevant to the query, then pull their neighbors.
        Uses THREE strategies, most reliable first:
          1. spaCy NER on the query
          2. Capitalized-word fallback
          3. Direct case-insensitive match against known graph node names
             (most reliable for casual/lowercase questions, since it checks
             against entities we KNOW exist rather than guessing from text)
        """
        doc = self.nlp(query)
        query_entities = [ent.text.strip() for ent in doc.ents]

        if not query_entities:
            query_entities = [w for w in query.split() if w[:1].isupper()]

        # Strategy 3: match query text against actual graph node names,
        # case-insensitively - catches cases like "ben" matching node "Ben"
        query_lower = query.lower()
        for node in self.graph.nodes():
            if node.lower() in query_lower and node not in query_entities:
                query_entities.append(node)

        facts = []
        for ent in query_entities:
            if ent in self.graph:
                for neighbor in self.graph.neighbors(ent):
                    edge_data = self.graph.get_edge_data(ent, neighbor)
                    sentence = edge_data.get("sentence", "")
                    facts.append(f'{ent} is related to {neighbor} (seen together in: "{sentence}")')

        return facts

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()
    
    def generate_html(self, path: str = "kg_store/graph.html") -> str:
        """Render the current graph as an interactive HTML file using pyvis."""
        from pyvis.network import Network
        import copy

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # pyvis serializes node/edge attributes to JSON, which can't handle
        # Python sets (used internally for "sources") - convert to a plain
        # list on a COPY of the graph so the original data model is untouched.
        graph_copy = copy.deepcopy(self.graph)
        for node, data in graph_copy.nodes(data=True):
            if "sources" in data and isinstance(data["sources"], set):
                data["sources"] = list(data["sources"])

        net = Network(height="550px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(graph_copy)
        net.write_html(path, open_browser=False, notebook=False)
        return path