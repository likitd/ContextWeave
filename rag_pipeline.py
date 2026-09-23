import time
import tiktoken
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

import config
from hybrid_retriever import (
    create_embedding_function,
    create_ensemble_retriever,
    load_existing_vector_store,
)
from agentfs_manager import AgentFSManager
from context_analysis import ContextAnalyzer

SYSTEM_PROMPT_AUTO = """\
You are a helpful assistant that answers questions using files from a context folder.
You have multiple retrieval strategies available as tools:

STRATEGY 1 — File exploration tools (list_directory, search_in_files):
  Use these when the query needs deep exploration of specific files, understanding
  code structure, tracing relationships across files, comparing content, or when
  the user asks about specific files/folders.

STRATEGY 2 — hybrid_search:
  Uses BM25 keyword matching + semantic vector search over pre-indexed chunks.
  Use this for straightforward factual questions, summaries, or when you need
  to quickly find relevant passages across many documents.

STRATEGY 3 — lightrag_search:
  Searches a LightRAG knowledge graph (entities + relations + chunks) using hybrid retrieval.
  Use for entity relationships, structured knowledge, or graph-aware context.

Decision guide:
- Simple factual Q&A, summaries, "what does X mean" → hybrid_search
- Entity relationships, structured queries → lightrag_search
- "Show me file X", "list the files", "what's in folder Y" → file tools
- "Find all occurrences of X", "grep for pattern" → search_in_files
- Complex multi-step analysis → combine hybrid_search and lightrag_search

Always cite which files your answer comes from.
"""

SYSTEM_PROMPT_LIGHTRAG = """\
You are a helpful assistant that answers questions using a LightRAG knowledge graph.
You have a lightrag_search tool that performs hybrid retrieval over entities, relations, and chunks.

Use lightrag_search for every question. It returns an answer grounded in the retrieved context.
Base your response on the tool result. Cite sources when possible. If you cannot find the answer, say so clearly.
"""

SYSTEM_PROMPT_AGENTFS_FILE_PICKER = """\
You are a file selection assistant. You receive a directory tree with file paths and keywords, plus a user request.
Your job is to decide which files are most relevant to fulfill the user request and how to batch them by importance.

Return one line per file. Each line: file_path or file_path|weight where weight is 1-10 (10 = most relevant).
Paths must match the overview exactly (e.g. folder/subfolder/file.txt). Use forward slashes.
Higher weight files will be read in earlier batches. Order lines by relevance if you like.
Include files that contain: examples, sample JSON, reference code, schemas, or anything that helps generate the requested output.
Do NOT include instructions or output format files — those are already provided separately.
Do NOT include any explanation. Output ONLY the list of paths (and optional |weight), one per line or comma-separated.
"""

SYSTEM_PROMPT_AGENTFS_MAIN = """\
You generate JSON (or other artifacts) using the provided instructions, schema, context, and user request.

The user message contains:
- [INSTRUCTIONS]: how to generate the output
- [OUTPUT FORMAT / SCHEMA]: the exact JSON schema to follow
- [AGENTFS CONTEXT]: reference material (examples, code, etc.) retrieved from a context folder
- [USER REQUEST]: what to generate

Using all of the above, generate the final output. Follow the instructions and schema exactly. Use the context as reference for structure and conventions.

RULES:
- Output the ACTUAL ARTIFACT (full JSON) only. No explanation, no summary, no markdown code fences.
- Follow the schema strictly. Use the retrieved context as reference for how the JSON should look.
"""

SYSTEM_PROMPT_AGENTFS_STRICT = """\
You answer questions using files from a context folder. You have these tools: get_context_overview, find_relevant_files, read_files_in_batches, search_in_files, process_all_files.

You MUST call get_context_overview first to obtain the directory tree and file paths. Only after you have the overview may you call find_relevant_files or search_in_files to narrow down, then read_files_in_batches to read content. Use the overview to choose valid file paths for read_files_in_batches.
Cite which files your answer comes from.
"""

SYSTEM_PROMPT_AGENTFS_FLEXIBLE = """\
You answer questions using files from a context folder. You have: get_context_overview (directory tree and paths), find_relevant_files (semantic file ranking), search_in_files (grep-like search), read_files_in_batches (read file content), process_all_files (read all relevant files).

Use the tools in whatever order best fits the question. For read_files_in_batches, pass file paths that appear in the overview or from find_relevant_files. Cite which files your answer comes from.
"""


def format_docs(docs: list[Document]) -> str:
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


class HybridRAGPipeline:
    def __init__(self):
        self.agentfs = AgentFSManager()
        self.context_analyzer = ContextAnalyzer()
        self.ensemble_retriever = None
        self.llm = None
        self.agent = None
        self.chunks: list[Document] = []
        self.subagent_token_usage = {"input": 0, "output": 0, "calls": 0}

    async def initialize(self):
        print("=" * 60)
        print("Initializing Hybrid RAG Pipeline")
        print("=" * 60)

        await self.agentfs.initialize()

        if config.STRATEGY in ("agentfs", "auto"):
            self.context_analyzer.build()

        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.API_KEY,
            base_url=config.BASE_URL or None,
        )

        if config.STRATEGY in ("hybrid", "auto"):
            embedding_fn = create_embedding_function()
            vector_store = load_existing_vector_store(embedding_fn)
            if vector_store:
                existing_docs = vector_store.get()
                self.chunks = [
                    Document(page_content=text, metadata=meta)
                    for text, meta in zip(existing_docs["documents"], existing_docs["metadatas"])
                ]
                self.ensemble_retriever = create_ensemble_retriever(
                    self.chunks, vector_store, embedding_fn
                )
                print(f"Loaded {len(self.chunks)} chunks from existing ChromaDB")
            else:
                print("WARNING: No existing ChromaDB found — run ingest.py first to index documents")

        if config.STRATEGY in ("lightrag", "auto"):
            from lightrag_manager import get_lightrag
            await get_lightrag()
            print("LightRAG retriever available (run lightrag_ingest.py to index)")

        tools = self._build_tools()
        prompts = {
            "agentfs": SYSTEM_PROMPT_AGENTFS_STRICT if config.AGENTFS_REQUIRE_OVERVIEW_FIRST else SYSTEM_PROMPT_AGENTFS_FLEXIBLE,
            "lightrag": SYSTEM_PROMPT_LIGHTRAG,
            "hybrid": SYSTEM_PROMPT_AUTO,
            "auto": SYSTEM_PROMPT_AUTO,
        }
        self.agent = create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=prompts.get(config.STRATEGY, SYSTEM_PROMPT_AUTO),
        )

        print("=" * 60)
        print(f"Pipeline ready!  (strategy={config.STRATEGY})")
        print("=" * 60)

    def _resolve_paths_in_context(self, paths: list[str]) -> list[str]:
        resolved = []
        for p in paths:
            normalized = p.strip().strip("/").strip("\\").replace("\\", "/")
            for candidate in [normalized, p.strip(), p.strip().replace("/", "\\")]:
                if not candidate:
                    continue
                if self.agentfs._read_file(candidate) is not None:
                    resolved.append(normalized if candidate == normalized else candidate.replace("\\", "/"))
                    break
        return list(dict.fromkeys(resolved))

    async def _read_files_batch(self, paths: list[str], question: str, path_weights: dict[str, float] | None = None) -> str:
        paths = self._resolve_paths_in_context(paths)
        if not paths:
            return "No valid files found. Check the paths."
        if path_weights is not None:
            paths = sorted(paths, key=lambda x: path_weights.get(x, 1.0), reverse=True)
        file_list = []
        for p in paths:
            content = self.agentfs._read_file(p)
            if content:
                file_list.append((p, content))
        if not file_list:
            return "No valid files found. Check the paths."
        enc_local = tiktoken.encoding_for_model("gpt-4o")
        max_tokens = config.SUBAGENT_MAX_TOKENS_PER_BATCH
        prompt_reserve = 100
        batches = []
        current_batch = []
        current_tokens = prompt_reserve
        for rel, content in file_list:
            file_tokens = len(enc_local.encode(content))
            if current_batch and current_tokens + file_tokens > max_tokens:
                batches.append(current_batch)
                current_batch = []
                current_tokens = prompt_reserve
            current_batch.append((rel, content))
            current_tokens += file_tokens
        if current_batch:
            batches.append(current_batch)
        print(f"\n  [SubAgent] Reading {len(file_list)} files in {len(batches)} batches...")
        sub_input_tokens = 0
        sub_output_tokens = 0
        sub_calls = 0
        batch_summaries = []
        for batch_idx, batch in enumerate(batches):
            print(f"Step/Process SubAgent LLM (batch {batch_idx + 1}/{len(batches)}) started...")
            file_contents = []
            for rel, content in batch:
                file_contents.append(f"=== FILE: {rel} ===\n{content}")
            combined = "\n\n".join(file_contents)
            sub_prompt = (
                f"USER QUESTION: {question}\n\n"
                f"Analyze the files below to answer the above question. "
                f"For each file, extract information relevant to the question. "
                f"Include file names, key details, and any data that helps answer it.\n\n{combined}"
            )
            sub_result = await self.llm.ainvoke([{"role": "user", "content": sub_prompt}])
            summary = sub_result.content
            batch_summaries.append(summary)
            sub_calls += 1
            batch_in = sub_result.usage_metadata.get("input_tokens", 0) if hasattr(sub_result, "usage_metadata") and sub_result.usage_metadata else 0
            batch_out = sub_result.usage_metadata.get("output_tokens", 0) if hasattr(sub_result, "usage_metadata") and sub_result.usage_metadata else 0
            sub_input_tokens += batch_in
            sub_output_tokens += batch_out
            print(f"  [SubAgent] Batch {batch_idx + 1}/{len(batches)} done ({len(batch)} files) — tokens: in={batch_in}, out={batch_out}")
        if len(batch_summaries) == 1:
            final = batch_summaries[0]
        else:
            print("Step/Process SubAgent LLM (merge) started...")
            merge_prompt = (
                f"USER QUESTION: {question}\n\n"
                f"Below are {len(batch_summaries)} batch analyses. Merge them into one coherent answer. Cite file names.\n\n"
                + "\n\n---\n\n".join(f"BATCH {i+1}:\n{s}" for i, s in enumerate(batch_summaries))
            )
            merge_result = await self.llm.ainvoke([{"role": "user", "content": merge_prompt}])
            final = merge_result.content
            sub_calls += 1
            merge_in = merge_result.usage_metadata.get("input_tokens", 0) if hasattr(merge_result, "usage_metadata") and merge_result.usage_metadata else 0
            merge_out = merge_result.usage_metadata.get("output_tokens", 0) if hasattr(merge_result, "usage_metadata") and merge_result.usage_metadata else 0
            sub_input_tokens += merge_in
            sub_output_tokens += merge_out
            print(f"  [SubAgent] Merge step done — tokens: in={merge_in}, out={merge_out}")
        print(f"  [SubAgent] Total sub-agent tokens — input: {sub_input_tokens}  |  output: {sub_output_tokens}  ({sub_calls} LLM calls)")
        self.subagent_token_usage["input"] += sub_input_tokens
        self.subagent_token_usage["output"] += sub_output_tokens
        self.subagent_token_usage["calls"] += sub_calls
        return final

    def _build_tools(self) -> list:
        agentfs = self.agentfs
        analyzer = self.context_analyzer
        ensemble_retriever = self.ensemble_retriever
        llm = self.llm
        pipeline_self = self
        enc = tiktoken.encoding_for_model("gpt-4o")

        def _log_output(name: str, out: str):
            print(f"  {name} output tokens (est.): {len(enc.encode(out))}")

        @tool
        async def get_context_overview() -> str:
            """Get the full directory tree with TF-IDF keywords per file.
            Call this FIRST to understand the file structure before reading any files."""
            call_id = await agentfs.tool_start("get_context_overview", {})
            overview = analyzer.get_graph_overview()
            await agentfs.tool_success(call_id, {"nodes": analyzer.graph.number_of_nodes()})
            _log_output("get_context_overview", overview)
            return overview

        @tool
        async def find_relevant_files(query: str, top_k: int = 15) -> str:
            """Optional: rank files by TF-IDF relevance to a query. Returns scored file list with keywords. Prefer choosing paths from [CONTEXT OVERVIEW] and using read_files_in_batches; use this only if you want a keyword-based search. Use top_k as integer, 15 or more."""
            k = int(top_k) if isinstance(top_k, str) else (top_k if isinstance(top_k, int) else 15)
            call_id = await agentfs.tool_start("find_relevant_files", {"query": query, "top_k": k})
            results = analyzer.find_relevant_files(query, k)
            await agentfs.tool_success(call_id, {"files_ranked": len(results)})
            if not results:
                out = "No files matched the query. Call process_all_files with the user's request to get context from all files in the folder."
            else:
                lines = [f"Top {len(results)} files for '{query}':"]
                for r in results:
                    kw = ", ".join(r["keywords"])
                    lines.append(f"  {r['file']}  (score: {r['score']}, keywords: {kw})")
                paths_csv = ",".join(r["file"] for r in results)
                lines.append(f"\nUse these paths in read_files_in_batches (copy exactly): {paths_csv}")
                out = "\n".join(lines)
            _log_output("find_relevant_files", out)
            return out

        @tool
        async def search_in_files(pattern: str) -> str:
            """Search for a text pattern (regex) across all files. Use only after read_files_in_batches if you need to find specific content."""
            call_id = await agentfs.tool_start("search_in_files", {"pattern": pattern})
            results = agentfs.search(pattern, max_results=10)
            await agentfs.tool_success(call_id, {"files_matched": len(results)})
            if not results:
                out = f"No matches found for pattern '{pattern}'."
            else:
                lines = []
                for r in results:
                    lines.append(f"\n{r['file']}:")
                    for m in r["matches"]:
                        lines.append(f"  L{m['line']}: {m['text']}")
                out = f"Search results for '{pattern}':" + "\n".join(lines)
            _log_output("search_in_files", out)
            return out

        async def _read_batch_impl(paths: list[str], question: str) -> str:
            return await pipeline_self._read_files_batch(paths, question)

        @tool
        async def read_files_in_batches(file_paths: str, question: str) -> str:
            """Read multiple files in token-budgeted batches to avoid sub-agent LLM context overflow. Pass a comma-separated list of file paths that you chose from [CONTEXT OVERVIEW] (each [FILE] line shows the full path). file_paths must not be empty. You may call again with more paths from the overview."""
            call_id = await agentfs.tool_start("read_files_in_batches", {"file_paths": file_paths, "question": question})
            paths = [p.strip().strip("/") for p in file_paths.replace("\n", ",").split(",") if p.strip()]
            paths = list(dict.fromkeys(paths))
            if len(paths) < 1:
                await agentfs.tool_success(call_id, {"files": 0})
                out = "Provide at least one file path."
                _log_output("read_files_in_batches", out)
                return out
            final = await _read_batch_impl(paths, question)
            await agentfs.tool_success(call_id, {"files": len(paths)})
            _log_output("read_files_in_batches", final)
            return final

        @tool
        async def process_all_files(question: str) -> str:
            """Process ALL files to answer a question that requires reading every document.
            Uses read_files_in_batches internally. Use for: full summaries, cross-file analysis,
            comparisons, pattern finding, aggregations."""
            call_id = await agentfs.tool_start("process_all_files", {"question": question})
            import os
            all_paths = analyzer.file_paths
            if not all_paths:
                await agentfs.tool_success(call_id, {"files": 0})
                out = "No files found in the context folder."
                _log_output("process_all_files", out)
                return out
            rel_paths = [os.path.relpath(fp, config.CONTEXT_FOLDER) for fp in all_paths]
            final = await _read_batch_impl(rel_paths, question)
            await agentfs.tool_success(call_id, {"files": len(rel_paths)})
            _log_output("process_all_files", final)
            return final

        @tool
        async def hybrid_search(query: str) -> str:
            """Search the knowledge base using hybrid BM25 keyword + semantic vector search.
            Returns the most relevant text passages from across all documents.
            Best for direct questions, summaries, or finding information by meaning."""
            call_id = await agentfs.tool_start("hybrid_search", {"query": query})
            if ensemble_retriever is None:
                await agentfs.tool_success(call_id, {"num_results": 0})
                out = "Hybrid search is not available — no documents were indexed."
            else:
                docs = ensemble_retriever.invoke(query)
                await agentfs.tool_success(call_id, {"num_results": len(docs)})
                out = "No relevant passages found." if not docs else format_docs(docs)
            _log_output("hybrid_search", out)
            return out

        @tool
        async def lightrag_search(query: str) -> str:
            """Search the LightRAG knowledge graph (entities, relations, chunks) using hybrid retrieval.
            Best for entity relationships, structured knowledge, and graph-aware context.
            Run lightrag_ingest.py first to index documents."""
            call_id = await agentfs.tool_start("lightrag_search", {"query": query})
            try:
                from lightrag_manager import query as lightrag_query
                out = await lightrag_query(query)
                await agentfs.tool_success(call_id, {"answer_length": len(out)})
            except Exception as e:
                await agentfs.tool_success(call_id, {"error": str(e)})
                out = f"LightRAG search failed: {e}. Run lightrag_ingest.py to index documents."
            _log_output("lightrag_search", out)
            return out

        agentfs_tools = [get_context_overview, find_relevant_files, read_files_in_batches, search_in_files, process_all_files]
        rag_tools = [hybrid_search]
        lightrag_tools = [lightrag_search]

        if config.STRATEGY == "agentfs":
            return agentfs_tools
        if config.STRATEGY == "hybrid":
            return rag_tools
        if config.STRATEGY == "lightrag":
            return lightrag_tools
        return agentfs_tools + rag_tools + lightrag_tools

    async def query(self, question: str) -> str:
        print(f"\nQuery: {question[:50]}... ({len(question)} characters)")
        print("-" * 50)

        if config.STRATEGY == "lightrag":
            call_id = await self.agentfs.tool_start("agent_query", {"question": question})
            lightrag_call_id = await self.agentfs.tool_start("lightrag_search", {"query": question})
            marker = "[USER REQUEST]"
            if marker in question:
                user_query = question.split(marker, 1)[1].strip()
            else:
                user_query = question
            print("Step/Process lightrag_search started...")
            print("Step/Process LightRAG query (retrieval) started...")
            from lightrag_manager import get_lightrag, query as lightrag_query
            rag = await get_lightrag()
            await rag.aclear_cache()
            lightrag_context = await lightrag_query(user_query)
            await self.agentfs.tool_success(lightrag_call_id, {"context_length": len(lightrag_context)})
            enc = tiktoken.encoding_for_model("gpt-4o")
            system_prompt = SYSTEM_PROMPT_AUTO
            final_user_content = f"{question}\n\n[LIGHTRAG CONTEXT]\n{lightrag_context}"
            input_tokens_est = len(enc.encode(system_prompt)) + len(enc.encode(final_user_content)) + 25
            print(f"Main agent input tokens (est.): {input_tokens_est}")
            print("Step/Process Agent (LLM) started...")
            result = await self.llm.ainvoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_user_content},
                ]
            )
            answer = result.content
            await self.agentfs.tool_success(call_id, {"answer_length": len(answer)})
            agent_input = 0
            agent_output = 0
            if hasattr(result, "usage_metadata") and result.usage_metadata:
                agent_input = result.usage_metadata.get("input_tokens", 0)
                agent_output = result.usage_metadata.get("output_tokens", 0)
            grand_input = agent_input
            grand_output = agent_output
            await self.agentfs.kv_set(
                f"query:{int(time.time())}",
                {
                    "question": question,
                    "answer": answer,
                    "tools_used": ["lightrag_search"],
                    "agent_tokens": {"input": agent_input, "output": agent_output},
                    "subagent_tokens": {"input": 0, "output": 0, "calls": 0},
                },
            )
            print(f"\nAnswer:\n{answer}\n")
            print(
                f"Agent tokens      →  input: {agent_input}  |  output: {agent_output}  |  total: {agent_input + agent_output}"
            )
            print(
                f"Grand total       →  input: {grand_input}  |  output: {grand_output}  |  total: {grand_input + grand_output}"
            )
            return answer

        if self.agent is None:
            return "Pipeline not initialized or no documents found."

        if config.STRATEGY == "agentfs":
            return await self._agentfs_query(question)

        self.subagent_token_usage = {"input": 0, "output": 0, "calls": 0}

        call_id = await self.agentfs.tool_start("agent_query", {"question": question})
        prompts = {"lightrag": SYSTEM_PROMPT_LIGHTRAG, "hybrid": SYSTEM_PROMPT_AUTO, "auto": SYSTEM_PROMPT_AUTO}
        system_prompt = prompts.get(config.STRATEGY, SYSTEM_PROMPT_AUTO)
        user_content = question
        enc = tiktoken.encoding_for_model("gpt-4o")
        input_tokens_est = len(enc.encode(system_prompt)) + len(enc.encode(user_content)) + 25
        print(f"Main agent input tokens (est.): {input_tokens_est}")
        print("Step/Process Agent (LLM) started...")
        result = await self.agent.ainvoke({"messages": [{"role": "user", "content": user_content}]})
        answer = result["messages"][-1].content
        await self.agentfs.tool_success(call_id, {"answer_length": len(answer)})

        agent_input = 0
        agent_output = 0
        tool_log = []
        for msg in result["messages"]:
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                agent_input += msg.usage_metadata.get("input_tokens", 0)
                agent_output += msg.usage_metadata.get("output_tokens", 0)
            if hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("args", {})
                    tool_log.append({"tool": name, "args": args})

        sub_in = self.subagent_token_usage["input"]
        sub_out = self.subagent_token_usage["output"]
        sub_calls = self.subagent_token_usage["calls"]
        grand_input = agent_input + sub_in
        grand_output = agent_output + sub_out

        await self.agentfs.kv_set(f"query:{int(time.time())}", {
            "question": question,
            "answer": answer,
            "tools_used": [t["tool"] for t in tool_log],
            "agent_tokens": {"input": agent_input, "output": agent_output},
            "subagent_tokens": {"input": sub_in, "output": sub_out, "calls": sub_calls},
        })

        print(f"\nAnswer:\n{answer}\n")
        print(f"Agent tokens      →  input: {agent_input}  |  output: {agent_output}  |  total: {agent_input + agent_output}")
        if sub_calls > 0:
            print(f"Sub-agent tokens  →  input: {sub_in}  |  output: {sub_out}  |  total: {sub_in + sub_out}  ({sub_calls} LLM calls)")
        print(f"Grand total       →  input: {grand_input}  |  output: {grand_output}  |  total: {grand_input + grand_output}")
        if tool_log:
            print("\nTool calls made by agent:")
            for i, t in enumerate(tool_log, 1):
                args_str = ", ".join(f"{k}={v!r}" for k, v in t["args"].items())
                print(f"  {i}. {t['tool']}({args_str})")
        return answer

    async def _agentfs_query(self, question: str) -> str:
        import os
        enc = tiktoken.encoding_for_model("gpt-4o")
        self.subagent_token_usage = {"input": 0, "output": 0, "calls": 0}
        call_id = await self.agentfs.tool_start("agent_query", {"question": question})

        overview_call_id = await self.agentfs.tool_start("get_context_overview", {})
        overview = self.context_analyzer.get_graph_overview()
        await self.agentfs.tool_success(overview_call_id, {"nodes": self.context_analyzer.graph.number_of_nodes()})
        print("Step 1/4: Context overview built")

        marker = "[USER REQUEST]"
        if marker in question:
            user_request_short = question.split(marker, 1)[1].strip()
        else:
            user_request_short = question

        picker_call_id = await self.agentfs.tool_start("file_picker_subagent", {"overview_len": len(overview)})
        picker_prompt = f"[CONTEXT OVERVIEW]\n{overview}\n\n[USER REQUEST]\n{user_request_short}"
        print("Step 2/4: Sub-agent picking relevant files from overview...")
        picker_result = await self.llm.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT_AGENTFS_FILE_PICKER},
            {"role": "user", "content": picker_prompt},
        ])
        raw_paths = picker_result.content.strip()
        picker_in = picker_result.usage_metadata.get("input_tokens", 0) if hasattr(picker_result, "usage_metadata") and picker_result.usage_metadata else 0
        picker_out = picker_result.usage_metadata.get("output_tokens", 0) if hasattr(picker_result, "usage_metadata") and picker_result.usage_metadata else 0
        self.subagent_token_usage["input"] += picker_in
        self.subagent_token_usage["output"] += picker_out
        self.subagent_token_usage["calls"] += 1
        await self.agentfs.tool_success(picker_call_id, {"raw_paths": raw_paths[:200]})

        path_weights = {}
        paths = []
        for part in raw_paths.replace("\n", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if "|" in part:
                p, w = part.rsplit("|", 1)
                p = p.strip().strip("/").strip("\\").replace("\\", "/")
                try:
                    path_weights[p] = float(w.strip())
                except ValueError:
                    path_weights[p] = 1.0
            else:
                p = part.strip().strip("/").strip("\\").replace("\\", "/")
                path_weights[p] = 1.0
            paths.append(p)
        paths = list(dict.fromkeys(paths))
        if not paths:
            all_rel = [os.path.relpath(fp, config.CONTEXT_FOLDER) for fp in self.context_analyzer.file_paths]
            paths = all_rel[:20]
            path_weights = {p: 1.0 for p in paths}
        paths_valid = self._resolve_paths_in_context(paths)
        print(f"  File picker selected {len(paths)} files, {len(paths_valid)} found in context")

        batch_call_id = await self.agentfs.tool_start("read_files_in_batches", {"files": len(paths_valid)})
        print(f"Step 3/4: Reading {len(paths_valid)} files in batches...")
        context_content = await self._read_files_batch(paths_valid, user_request_short, path_weights if path_weights else None)
        await self.agentfs.tool_success(batch_call_id, {"files": len(paths_valid)})

        main_call_id = await self.agentfs.tool_start("main_llm_generation", {})
        final_user_content = (
            f"{question}\n\n"
            f"[AGENTFS CONTEXT]\n{context_content}"
        )
        main_input_est = len(enc.encode(SYSTEM_PROMPT_AGENTFS_MAIN)) + len(enc.encode(final_user_content)) + 25
        print(f"Step 4/4: Main LLM generating final output (est. {main_input_est} input tokens)...")
        main_result = await self.llm.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT_AGENTFS_MAIN},
            {"role": "user", "content": final_user_content},
        ])
        answer = main_result.content
        await self.agentfs.tool_success(main_call_id, {"answer_length": len(answer)})
        await self.agentfs.tool_success(call_id, {"answer_length": len(answer)})

        main_in = main_result.usage_metadata.get("input_tokens", 0) if hasattr(main_result, "usage_metadata") and main_result.usage_metadata else 0
        main_out = main_result.usage_metadata.get("output_tokens", 0) if hasattr(main_result, "usage_metadata") and main_result.usage_metadata else 0
        sub_in = self.subagent_token_usage["input"]
        sub_out = self.subagent_token_usage["output"]
        sub_calls = self.subagent_token_usage["calls"]
        grand_input = main_in + sub_in
        grand_output = main_out + sub_out

        await self.agentfs.kv_set(f"query:{int(time.time())}", {
            "question": question,
            "answer": answer,
            "tools_used": ["file_picker_subagent", "read_files_in_batches", "main_llm_generation"],
            "agent_tokens": {"input": main_in, "output": main_out},
            "subagent_tokens": {"input": sub_in, "output": sub_out, "calls": sub_calls},
        })

        print(f"\nAnswer:\n{answer}\n")
        print(f"Main LLM tokens   →  input: {main_in}  |  output: {main_out}  |  total: {main_in + main_out}")
        print(f"Sub-agent tokens  →  input: {sub_in}  |  output: {sub_out}  |  total: {sub_in + sub_out}  ({sub_calls} LLM calls)")
        print(f"Grand total       →  input: {grand_input}  |  output: {grand_output}  |  total: {grand_input + grand_output}")
        return answer

    async def get_pipeline_stats(self):
        stats = await self.agentfs.get_stats()
        print("\n--- Pipeline Stats (AgentFS Tool Calls) ---")
        for stat in stats:
            print(f"  {stat.name}: {stat.total_calls} calls, {stat.successful} ok, avg {stat.avg_duration_ms:.1f}ms")
        return stats

    async def shutdown(self):
        await self.agentfs.close()
