# Section 2A: AI/LLM Engineering - ReACT Agent, Anomaly Detection, Ticket Triaging

## Detailed Interview Reference Answers - Deepaksakthi (SuperOps)

---

## Part 1: ReACT Agent for Automated First Responses (Q21-Q30)

---

### Q21. Walk me through the ReACT agent loop step by step. How does Reasoning + Acting work in your ticket first-response system?

The ReACT (Reasoning + Acting) pattern is an iterative loop where the LLM alternates between thinking about what to do and taking concrete actions via tools. In our ticket first-response system at SuperOps, the loop worked as follows:

**Step 1 - Observation (Ticket Intake):**
When a new support ticket arrives (via Slack, Teams, or Email), the system extracts the raw text, metadata (channel, customer ID, priority hints, attachments), and passes it as the initial observation to the agent.

**Step 2 - Reasoning (Thought):**
The LLM analyzes the ticket content and generates a "Thought" — an internal reasoning step. For example: *"The user is reporting that their backup agent on Windows Server 2019 is failing with error code 0x8007045D. I should look up this error code in our knowledge base and check if there are known resolutions."*

**Step 3 - Action (Tool Call):**
Based on the reasoning, the agent selects and invokes a tool. For instance, it calls `search_knowledge_base(query="backup agent error 0x8007045D Windows Server 2019")`.

**Step 4 - Observation (Tool Result):**
The tool returns results — say, three KB articles about this error relating to disk I/O issues and a known fix involving driver updates.

**Step 5 - Reasoning (Refined Thought):**
The agent reasons again: *"The KB articles suggest this is a disk I/O error. I should also check the customer's asset profile to see if there are recent disk health alerts."*

**Step 6 - Action (Second Tool Call):**
The agent calls `get_asset_health(customer_id="C-1234", asset_type="server")`.

**Step 7 - Observation and Final Reasoning:**
After gathering sufficient context, the agent reasons that it has enough information to draft a response.

**Step 8 - Final Action (Generate Response):**
The agent produces a structured first response that includes: acknowledgment of the issue, likely root cause, recommended steps, and relevant KB links.

Here is a simplified implementation:

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1)

react_prompt = """You are a Level-1 IT support agent for SuperOps.
Answer the following support ticket using the tools available.

Ticket: {input}
Channel: {channel}
Customer Context: {customer_context}

{agent_scratchpad}

Use this format:
Thought: <your reasoning>
Action: <tool name>
Action Input: <tool input>
Observation: <tool result>
... (repeat as needed)
Thought: I have enough information to respond.
Final Answer: <your response to the customer>
"""

agent = create_react_agent(llm, tools, react_prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,       # Cap loops to prevent runaway
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=True  # For audit trail
)
```

The key design decision was capping iterations at 5. Most tickets resolved in 2-3 loops. If the agent hit 5 iterations without a final answer, we triggered the fallback (escalation to a human). We also logged every intermediate step for auditing and continuous improvement.

---

### Q22. What tools did you give the ReACT agent? How did it decide which tool to use?

We equipped the agent with a curated set of five core tools, each with a clear description that guided the LLM's tool selection:

**1. `search_knowledge_base`** — Performed semantic search over our internal KB articles, runbooks, and documentation using vector embeddings stored in a vector database.

**2. `get_asset_info`** — Retrieved real-time and historical information about a customer's assets (servers, endpoints, network devices) including recent alerts, health status, and configuration.

**3. `search_similar_tickets`** — Found historically resolved tickets similar to the current one, returning the resolution steps that worked.

**4. `get_customer_context`** — Pulled customer account details: SLA tier, product subscriptions, past interactions, and escalation preferences.

**5. `check_service_status`** — Checked the status of SuperOps services and known outages that might explain the reported issue.

```python
tools = [
    Tool(
        name="search_knowledge_base",
        func=kb_search,
        description=(
            "Search the internal knowledge base for troubleshooting guides, "
            "error codes, and resolution steps. Input should be a concise "
            "search query describing the technical issue."
        )
    ),
    Tool(
        name="get_asset_info",
        func=asset_lookup,
        description=(
            "Get real-time health, configuration, and recent alerts for a "
            "customer's asset. Input should be JSON with 'customer_id' and "
            "optionally 'asset_id' or 'asset_type'."
        )
    ),
    Tool(
        name="search_similar_tickets",
        func=similar_ticket_search,
        description=(
            "Find previously resolved tickets similar to the current issue. "
            "Returns resolution steps that worked. Input should be a "
            "description of the issue in plain text."
        )
    ),
    # ... similar for others
]
```

**How did the agent decide which tool to use?**

The LLM selected tools based on three factors:

1. **Tool descriptions:** We spent significant effort crafting precise, unambiguous descriptions. The description acts as the tool's "contract" — the LLM matches the current need against available descriptions. We iterated on these descriptions extensively during development; vague descriptions led to wrong tool choices.

2. **Few-shot examples in the prompt:** We included 2-3 example reasoning traces in the system prompt showing the agent choosing different tools for different ticket types. This grounded the agent's behavior.

3. **Contextual reasoning:** The LLM naturally reasons about what information it needs. If the ticket mentions an error code, it leans toward `search_knowledge_base`. If it references "my server" or "my endpoint," it reaches for `get_asset_info` first.

We tracked tool selection accuracy in production and found that the agent chose the correct first tool ~92% of the time. The remaining 8% usually self-corrected in the next iteration — the agent would see an unhelpful result and pivot to a better tool.

---

### Q23. How did you handle hallucinations in the agent's responses? What guardrails did you implement?

Hallucination was our biggest concern since we were generating customer-facing responses. We implemented a multi-layered guardrail system:

**Layer 1 - Grounded Generation (Architectural):**
The fundamental defense was forcing the agent to use tools before answering. The system prompt explicitly stated: *"Never answer from your own knowledge. Always verify information using the available tools. If no tool returns relevant information, say you are escalating."* This meant every factual claim in the response should trace back to a tool result.

**Layer 2 - Source Attribution Check:**
After the agent generated a final response, a lightweight validation step checked whether the response contained claims that could not be traced to any tool observation in the intermediate steps. We implemented this as a second LLM call with a focused prompt:

```python
validation_prompt = """
Given the following tool results and the agent's draft response, 
identify any claims in the response that are NOT supported by the 
tool results. Return a JSON with:
- "supported": true/false
- "unsupported_claims": [list of unsupported statements]

Tool Results: {tool_observations}
Draft Response: {agent_response}
"""

validation = await llm.ainvoke(validation_prompt)
if not validation["supported"]:
    response = apply_hedging(response, validation["unsupported_claims"])
```

**Layer 3 - Confidence Scoring:**
We asked the agent to self-assess confidence (low/medium/high) as part of its final reasoning step. Low-confidence responses were routed to human review before sending. We calibrated this by comparing self-assessed confidence against human evaluations over a test set.

**Layer 4 - Blocklist and Regex Filters:**
Post-processing filters caught dangerous patterns: promises of refunds or SLA credits, definitive statements about data loss, security vulnerability disclosures, or mentions of competitor products. These triggered automatic escalation.

**Layer 5 - Template Anchoring:**
For common ticket categories (password resets, license questions, known bugs), we had pre-approved response templates. The agent was instructed to use these templates and fill in specifics rather than generating entirely freeform responses.

**Layer 6 - Human-in-the-Loop for High-Risk Categories:**
Tickets classified as billing, security, or data-related always went through human review regardless of agent confidence.

The combination reduced hallucination-related incidents to under 2% of auto-responses in production, measured by weekly human audits of a random sample.

---

### Q24. How did the agent handle multi-channel context (Slack vs Teams vs Email)? Did the prompt change per channel?

Yes, the prompt and post-processing adapted per channel, though the core reasoning loop remained the same. We handled this through a channel adapter pattern:

**1. Channel-Specific Prompt Injection:**
A `channel_context` block was injected into the prompt that adjusted tone, formatting, and length:

```python
CHANNEL_CONFIGS = {
    "slack": {
        "tone": "conversational and concise",
        "format": "Use short paragraphs. Use Slack markdown (bold with *, code with `). Use bullet points. Keep under 300 words.",
        "greeting": "Keep greeting casual: 'Hi {name}!'",
        "sign_off": "End with a friendly note and emoji if appropriate."
    },
    "teams": {
        "tone": "professional but approachable",
        "format": "Use Teams-compatible markdown. Structure with headers if multi-step. Keep under 400 words.",
        "greeting": "Use professional greeting: 'Hello {name},'",
        "sign_off": "End with 'Best regards' and agent name."
    },
    "email": {
        "tone": "formal and thorough",
        "format": "Use proper email structure with greeting, body paragraphs, numbered steps for instructions, and signature block. Can be up to 500 words.",
        "greeting": "Use formal greeting: 'Dear {name},'",
        "sign_off": "Include full signature block with support contact info."
    }
}
```

**2. Input Normalization:**
Each channel had different raw input formats. Slack messages included thread context and emoji reactions. Teams messages had adaptive card metadata. Emails had subject lines, CC lists, and reply chains. We built channel-specific parsers that normalized all of this into a unified `TicketContext` object:

```python
class TicketContext:
    raw_text: str
    subject: Optional[str]        # Email only
    thread_history: List[str]     # Slack/Teams threads
    sender_name: str
    sender_email: str
    channel: str                  # slack | teams | email
    attachments: List[Attachment]
    cc_recipients: List[str]      # Email only
    urgency_signals: List[str]    # e.g., "URGENT" in subject, red emoji
```

**3. Output Formatting:**
The agent's raw response went through a channel-specific formatter before delivery. For Slack, we converted markdown to Slack Block Kit JSON. For Teams, we generated Adaptive Cards for structured responses (like step-by-step guides). For Email, we wrapped the response in an HTML template.

**4. Context Window Differences:**
Slack threads could contain 50+ messages of back-and-forth. We used a summarization step for long threads before passing to the agent, keeping the most recent 5 messages verbatim and summarizing earlier context. Email chains were similarly truncated using a "most-recent-first" approach.

The channel itself also influenced tool selection implicitly: email tickets tended to be more detailed and formal (thus more likely to be complex issues), while Slack messages were often quick questions that the agent could resolve in fewer iterations.

---

### Q25. What was the fallback mechanism when the agent couldn't generate a confident response?

We designed a graduated fallback system with three tiers:

**Tier 1 - Graceful Acknowledgment (Automatic):**
If the agent completed its reasoning loop but self-assessed low confidence, it generated a response that acknowledged the issue, confirmed what it understood, and set expectations:

```
"Hi [Name], thanks for reaching out. I can see you're experiencing [issue summary]. 
I want to make sure we get this right, so I'm connecting you with a specialist 
who can help. In the meantime, could you confirm [clarifying question]? 
A team member will follow up within [SLA time]."
```

This was still auto-sent but clearly set the expectation that a human would follow up.

**Tier 2 - Human Review Queue (Semi-Automatic):**
If the agent hit max iterations (5) without converging on a final answer, or if the validation layer flagged unsupported claims, the draft response was routed to a human review queue. The human agent saw:
- The original ticket
- The agent's intermediate reasoning steps
- The draft response (if any)
- A suggested classification and priority

This reduced human effort by ~40% compared to handling from scratch since the agent had already gathered relevant context.

**Tier 3 - Direct Escalation (Immediate):**
Certain trigger conditions bypassed the agent entirely:
- Ticket classified as security incident or data breach
- Customer flagged as VIP/enterprise tier
- Sentiment analysis detected extreme frustration or legal language
- Keywords matched critical patterns ("lawsuit," "data leak," "compliance violation")

```python
async def process_ticket(ticket: TicketContext) -> Response:
    # Check Tier 3 triggers first
    if should_escalate_immediately(ticket):
        return create_escalation(ticket, reason="auto_escalate_trigger")
    
    # Run the agent
    result = await agent_executor.ainvoke({
        "input": ticket.raw_text,
        "channel": ticket.channel,
        "customer_context": ticket.customer_context
    })
    
    # Check confidence and validation
    confidence = extract_confidence(result)
    validation = await validate_response(result)
    
    if confidence == "high" and validation["supported"]:
        return auto_send(result["output"], ticket)        # Tier 0: Auto-send
    elif confidence == "medium" and validation["supported"]:
        return auto_send_with_flag(result["output"], ticket)  # Auto-send, flag for review
    elif confidence == "low" or not validation["supported"]:
        return route_to_human_queue(result, ticket)       # Tier 2
    else:
        return send_acknowledgment(ticket)                # Tier 1
```

We tracked fallback rates as a key metric. Initially ~35% of tickets hit Tier 1 or 2. After three months of prompt tuning, KB expansion, and tool improvement, this dropped to ~18%.

---

### Q26. How did you evaluate the quality of agent-generated first responses? What metrics did you track?

We used a combination of automated metrics, human evaluation, and business outcome metrics:

**Automated Metrics:**

1. **Resolution Rate (First-Response Resolution):** Percentage of tickets where the customer's issue was resolved by the agent's first response alone, with no further human follow-up needed. This was our north-star metric. We started at ~28% and reached ~45% after tuning.

2. **Fallback Rate:** Percentage of tickets that triggered Tier 1/2 fallbacks. Lower is better, but we also tracked false confidence — cases where the agent was confident but wrong.

3. **Latency:** P50, P95, and P99 end-to-end response times. Target was under 30 seconds for P95.

4. **Tool Utilization:** How many tools were called per ticket, which tools were used, and whether tool results were actually incorporated into the response.

**Human Evaluation (Weekly Audit):**

We sampled 100 agent responses per week and had senior support engineers rate them on a 5-point scale across four dimensions:

- **Accuracy:** Is the information factually correct? (Most critical)
- **Completeness:** Does it address all aspects of the customer's question?
- **Tone:** Is it appropriate for the channel and customer context?
- **Actionability:** Does it give the customer clear next steps?

```python
# Evaluation tracking
class ResponseEvaluation:
    ticket_id: str
    accuracy_score: int       # 1-5
    completeness_score: int   # 1-5
    tone_score: int           # 1-5
    actionability_score: int  # 1-5
    hallucination_detected: bool
    would_human_send_as_is: bool  # Key binary metric
    reviewer_notes: str
```

The **"would send as-is"** metric was particularly telling — it measured whether a human agent would have sent the response without modifications. We started at 52% and improved to 73%.

**Business Outcome Metrics:**

- **Mean Time to First Response (MTFR):** Dropped from ~45 minutes (human) to ~20 seconds (agent).
- **Customer Satisfaction (CSAT):** Tracked per-ticket CSAT scores for agent vs human responses. Agent responses initially scored 3.6/5 vs human 4.2/5. After improvements, agent reached 4.0/5.
- **Escalation Rate:** Percentage of agent-handled tickets that eventually required human intervention anyway. This differed from fallback rate — it captured cases where the agent responded but the customer came back unsatisfied.
- **Agent Productivity:** Human agents' tickets-per-day increased as the AI handled routine queries.

We built a dashboard that displayed all these metrics with weekly trends, and held bi-weekly review sessions to identify patterns in failures and drive improvements.

---

### Q27. What was the latency of the agent pipeline end-to-end? How did you optimize it?

**Baseline Latency Breakdown:**

| Stage | P50 | P95 |
|-------|-----|-----|
| Input parsing & normalization | 50ms | 100ms |
| Customer context fetch | 80ms | 200ms |
| LLM reasoning (per iteration) | 1.2s | 3.5s |
| Tool execution (per call) | 200ms | 800ms |
| Response validation | 800ms | 2s |
| Output formatting | 30ms | 80ms |
| **Total (2-3 iterations)** | **~4s** | **~12s** |

Our target was P95 under 15 seconds for the full pipeline. Initial P95 was around 18 seconds. Here is what we did to optimize:

**1. Parallel Context Fetching:**
Before the agent loop even started, we fetched customer context, recent tickets, and asset info in parallel using `asyncio.gather()`. This pre-populated context reduced the need for tool calls during the loop.

```python
async def prepare_context(ticket: TicketContext):
    customer_ctx, recent_tickets, asset_info = await asyncio.gather(
        get_customer_context(ticket.sender_email),
        get_recent_tickets(ticket.sender_email, limit=5),
        get_asset_health(ticket.customer_id),
    )
    return EnrichedContext(customer_ctx, recent_tickets, asset_info)
```

**2. Streaming with Early Termination:**
We used streaming responses from the LLM API. For the validation step, we used a smaller, faster model (GPT-3.5-turbo) instead of GPT-4, since validation is a simpler task.

**3. Semantic Caching:**
We implemented a semantic cache using embeddings. If an incoming ticket was highly similar (cosine similarity > 0.95) to a recently processed ticket, we served the cached response (with customer-specific details swapped). This gave near-instant responses for common recurring issues. The cache hit rate was approximately 15%.

```python
class SemanticCache:
    def __init__(self, embedding_model, threshold=0.95):
        self.cache = {}  # embedding -> response
        self.threshold = threshold
    
    async def get(self, ticket_text: str) -> Optional[str]:
        embedding = await self.embedding_model.embed(ticket_text)
        for cached_emb, cached_response in self.cache.items():
            if cosine_similarity(embedding, cached_emb) > self.threshold:
                return cached_response
        return None
```

**4. Model Selection Strategy:**
We used a two-tier model approach. Simple tickets (password resets, status checks) were routed to GPT-3.5-turbo which was 5-8x faster. Complex tickets went to GPT-4-turbo. A lightweight classifier made this routing decision in ~100ms.

**5. Tool Response Caching:**
KB search results and asset info were cached with a 5-minute TTL in Redis, avoiding redundant database and API calls.

After all optimizations, our final latency was P50: ~3s, P95: ~10s, P99: ~18s.

---

### Q28. How did you handle rate limits and token limits with the LLM API in a production ticket system?

**Rate Limit Handling:**

1. **Token Bucket with Backpressure:**
We implemented a token bucket rate limiter that sat in front of all LLM API calls. When nearing the rate limit, we applied backpressure — tickets were queued rather than dropped, and we adjusted processing concurrency dynamically.

```python
class LLMRateLimiter:
    def __init__(self, rpm_limit=500, tpm_limit=150000):
        self.request_bucket = TokenBucket(rpm_limit, refill_period=60)
        self.token_bucket = TokenBucket(tpm_limit, refill_period=60)
        self.queue = asyncio.PriorityQueue()
    
    async def call_llm(self, prompt: str, priority: int = 5):
        estimated_tokens = len(prompt) // 4 + 500  # rough estimate
        
        # Wait for both request and token capacity
        await self.request_bucket.acquire(1)
        await self.token_bucket.acquire(estimated_tokens)
        
        try:
            return await self._make_api_call(prompt)
        except RateLimitError:
            await asyncio.sleep(calculate_backoff())
            return await self._make_api_call(prompt)
```

2. **Priority Queuing:**
High-priority tickets (P1 incidents, VIP customers) got priority in the LLM request queue. Low-priority tickets could tolerate slightly higher latency.

3. **Multi-Provider Failover:**
We had Azure OpenAI as a secondary provider. If the primary hit rate limits or experienced degradation, we automatically failed over. Both providers were behind a unified abstraction layer.

**Token Limit Management:**

1. **Context Window Budgeting:**
We allocated the context window (8K/32K tokens depending on model) into explicit budgets:
   - System prompt: ~1500 tokens (fixed)
   - Ticket + customer context: ~2000 tokens (variable, truncated if needed)
   - Tool results: ~2000 tokens (summarized if individual results were too long)
   - Reasoning history: ~1500 tokens (sliding window of recent steps)
   - Response generation: ~1000 tokens (reserved for output)

2. **Intelligent Truncation:**
Long email threads and KB articles were summarized before injection. We used a fast extractive summarization approach — pulling the most relevant sentences based on TF-IDF similarity to the ticket query rather than running another LLM call.

3. **Sliding Window for Agent Scratchpad:**
For tickets requiring many iterations, older reasoning steps were compressed. We kept the most recent 2 thought-action-observation triples in full and summarized earlier ones into a compact "context so far" block.

4. **Token Usage Monitoring:**
We tracked token consumption per ticket category, per model, and per agent tool. This let us identify and optimize the most token-expensive ticket types. Monthly LLM API costs were a key operational metric we reported.

---

### Q29. Did you use function calling or tool-use APIs? How did you structure the function schemas?

Yes, we migrated from the text-based ReACT parsing (where the LLM outputs "Action: tool_name") to OpenAI's native function calling API. This was a significant reliability improvement — structured function calling eliminated parsing failures that occurred with the text-based approach roughly 5-8% of the time.

**Function Schema Design:**

Each tool was defined as a JSON Schema compatible function. Here is an example:

```python
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the internal knowledge base for troubleshooting "
                "guides, error resolutions, and product documentation. "
                "Use this when the ticket mentions a specific error, "
                "feature question, or how-to request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Concise search query describing the issue"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["error_resolution", "how_to", "product_docs", "release_notes"],
                        "description": "Category to narrow the search"
                    },
                    "product": {
                        "type": "string",
                        "enum": ["rmm", "psa", "backup", "security", "general"],
                        "description": "Product area if identifiable from the ticket"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_asset_info",
            "description": (
                "Retrieve health status, configuration, and recent alerts "
                "for a customer's managed asset. Use this when the ticket "
                "references a specific device, server, or endpoint."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer identifier"
                    },
                    "asset_id": {
                        "type": "string",
                        "description": "Specific asset ID if known"
                    },
                    "asset_type": {
                        "type": "string",
                        "enum": ["server", "workstation", "network_device", "all"],
                        "description": "Type of asset to query"
                    },
                    "include_alerts": {
                        "type": "boolean",
                        "description": "Whether to include recent alerts (last 7 days)",
                        "default": True
                    }
                },
                "required": ["customer_id"]
            }
        }
    }
]
```

**Key Design Principles:**

1. **Descriptive `enum` values:** Instead of free-text parameters, we used enums wherever possible. This constrained the LLM's output space and prevented invalid inputs.

2. **Clear `description` at every level:** Every parameter had a description explaining not just what it is, but when to use it. The function-level description explained the tool's purpose and trigger conditions.

3. **Minimal required parameters:** Only truly essential parameters were marked required. Optional parameters had sensible defaults, reducing the cognitive load on the LLM.

4. **Consistent naming conventions:** All tools used snake_case, all IDs were strings, and all boolean flags followed the `include_X` or `filter_by_X` pattern.

We used LangChain's `convert_to_openai_function` utility for tools defined as Python functions with type hints, which auto-generated the schema. But for production, we hand-tuned the descriptions and added enums that auto-generation missed.

---

### Q30. How did you manage prompt versioning and A/B testing for the agent?

**Prompt Versioning:**

We treated prompts as code artifacts, versioned in Git alongside the application code. Each prompt had:

```
prompts/
  react_agent/
    v1.0.0_system.txt
    v1.1.0_system.txt
    v1.2.0_system.txt
    channel_configs.yaml
    few_shot_examples.yaml
  validation/
    v1.0.0_validator.txt
  classification/
    v1.0.0_classifier.txt
```

Each prompt version included a header comment documenting the change and the evaluation results that justified the change:

```
# Version: 1.2.0
# Date: 2024-03-15
# Change: Added explicit instruction to check service status before 
#   suggesting troubleshooting steps. Reduced "try restarting" responses 
#   during outages by 60%.
# Eval: accuracy 4.2->4.3, completeness 3.8->4.1 on 200-ticket test set
```

We used a configuration service (backed by a database with a Redis cache) that mapped prompt version to deployment environment:

```python
class PromptManager:
    def __init__(self, config_service):
        self.config = config_service
    
    async def get_prompt(self, prompt_name: str, context: dict) -> str:
        # Get version assignment (could be A/B test)
        version = await self.config.get_prompt_version(
            prompt_name=prompt_name,
            customer_id=context.get("customer_id"),
            channel=context.get("channel")
        )
        prompt_template = await self.load_prompt(prompt_name, version)
        return prompt_template.format(**context)
```

**A/B Testing Framework:**

We built a lightweight A/B testing system specifically for prompts:

1. **Traffic Splitting:** We used consistent hashing on `customer_id` to ensure the same customer always got the same prompt version (avoiding confusion from inconsistent response styles). Splits were configurable (e.g., 80/20 or 50/50).

2. **Metric Collection:** Every response was tagged with the prompt version. We tracked all our standard metrics (accuracy, resolution rate, CSAT, latency) segmented by prompt version.

3. **Statistical Significance:** We used a Bayesian approach to determine when we had enough data to declare a winner. For our volume (~500 tickets/day), we typically needed 1-2 weeks to reach significance on resolution rate.

4. **Guardrails on A/B Tests:** New prompt versions were always tested against our offline evaluation set (200 curated tickets with human-labeled ideal responses) before going to production A/B. They needed to match or exceed the baseline on accuracy before being allowed into the A/B test.

5. **Rollback:** If a new prompt version showed degradation on any critical metric during A/B testing, we had a one-click rollback in our admin dashboard that instantly set the traffic split to 100% baseline.

A typical prompt improvement cycle looked like: identify failure pattern in weekly audits, hypothesize a prompt fix, test offline, deploy as 20% A/B, monitor for 1-2 weeks, promote to 100% or roll back.

---

## Part 2: Anomaly Detection on Asset Time Series Data (Q31-Q40)

---

### Q31. What statistical models did you use for anomaly detection on asset time series data? Why not deep learning?

We used a combination of statistical methods, chosen deliberately over deep learning:

**Primary Models:**

1. **Seasonal-Trend Decomposition with LOESS (STL) + Residual Thresholding:**
This was our workhorse. We decomposed each time series into trend, seasonal, and residual components. Anomalies were detected in the residual component using a modified Z-score (MAD-based rather than standard deviation, for robustness to outliers).

```python
from statsmodels.tsa.seasonal import STL

def detect_anomalies_stl(series: pd.Series, period: int = 24) -> List[Anomaly]:
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    
    residuals = result.resid
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    modified_z_scores = 0.6745 * (residuals - median) / mad
    
    anomalies = np.abs(modified_z_scores) > 3.5
    return anomalies
```

2. **Exponential Weighted Moving Average (EWMA) Control Charts:**
For real-time streaming detection, we used EWMA with dynamically computed control limits. This was particularly effective for detecting gradual drift (e.g., slow memory leak).

3. **Isolation Forest:**
For multivariate anomaly detection (e.g., CPU + memory + disk I/O together), we used Isolation Forest. It handles high-dimensional data well and does not assume a distribution.

4. **Prophet (for capacity planning alerts):**
We used Facebook Prophet for longer-horizon forecasting — predicting when a metric would breach a threshold in the next 7-30 days (e.g., disk will be full in 12 days).

**Why Not Deep Learning?**

1. **Data volume per asset:** Each individual asset had at most weeks to months of data. Deep learning models (LSTMs, Transformers) need large training sets. We had thousands of assets but each had limited history — and transfer learning across different asset types is unreliable because baselines differ drastically.

2. **Interpretability:** When we fired an alert, MSP technicians needed to understand why. "The STL residual exceeded 3.5 MAD" is debuggable. "The autoencoder reconstruction error was 0.87" is not. Interpretability was a hard requirement from customers.

3. **Operational simplicity:** Statistical models are deterministic, require no GPU, have predictable latency, and are easy to unit test. We were running detection on tens of thousands of time series — the operational overhead of managing deep learning model training, GPU allocation, and model versioning would have been disproportionate.

4. **Diminishing returns:** In our evaluation (detailed in Q38), statistical methods achieved 91% precision and 87% recall. We prototyped an LSTM-based detector and it achieved 92% precision and 89% recall — a marginal improvement that did not justify the complexity.

5. **Cold start:** Statistical methods can start working with as little as 2-3 periods of seasonal data. Deep learning needs much more.

The principle was: use the simplest method that meets the accuracy bar, and invest engineering effort in the parts that matter more — data pipeline reliability, alert routing, and reducing false positives.

---

### Q32. How did you define "anomaly" — point anomalies, contextual anomalies, or collective anomalies? Give examples from your system.

We handled all three types, and the distinction was critical for generating actionable alerts:

**1. Point Anomalies:**
A single data point that is significantly different from the rest of the data, regardless of context.

*Example:* CPU usage on a server suddenly spikes to 99% when its normal range is 10-40%. This is a straightforward outlier detected by our Z-score thresholding on STL residuals.

*Detection method:* STL decomposition + modified Z-score. If the residual exceeds 3.5 MAD, it is a point anomaly.

**2. Contextual (Conditional) Anomalies:**
A data point that is anomalous in a specific context but would be normal in another.

*Example:* 80% CPU usage at 3 AM is anomalous for an office workstation (expected to be near-idle), but perfectly normal for a backup server that runs nightly backup jobs at 2 AM. Similarly, 90% CPU on a Monday morning might be normal (users logging in) but anomalous on a Saturday.

*Detection method:* This is where our seasonality handling became crucial. By decomposing the signal with STL using appropriate seasonal periods (hourly within a day, daily within a week), the seasonal component captured expected patterns. The residual-based detection then only flagged deviations from the expected seasonal behavior. We also maintained per-asset "behavioral profiles" that encoded known scheduled tasks.

```python
class AssetBehaviorProfile:
    scheduled_tasks: List[ScheduledTask]  # e.g., backup at 2AM
    business_hours: Tuple[int, int]       # e.g., (8, 18)
    expected_idle_periods: List[TimeRange]
    
    def is_expected_high_usage(self, timestamp: datetime, metric: str) -> bool:
        for task in self.scheduled_tasks:
            if task.overlaps(timestamp) and task.affects_metric(metric):
                return True
        return False
```

**3. Collective Anomalies:**
A sequence of data points that together constitute an anomaly, even though individual points might not be anomalous.

*Example:* Disk I/O latency gradually increasing from 5ms to 25ms over 6 hours. No single data point is a dramatic outlier, but the persistent upward trend signals a degrading disk. Another example: network packet loss fluctuating between 0-2% normally, but showing a sustained 3-5% for several hours — each point is within a plausible range, but the sustained elevation is anomalous.

*Detection method:* We used EWMA control charts for drift detection and a sliding window approach that tracked the proportion of "borderline" readings. If more than 60% of readings in a 2-hour window fell above the 90th percentile of the historical distribution, we flagged a collective anomaly.

```python
def detect_collective_anomaly(series: pd.Series, window: str = "2h",
                               threshold_percentile: float = 0.9,
                               proportion_threshold: float = 0.6) -> bool:
    threshold = series.quantile(threshold_percentile)
    window_data = series.last(window)
    proportion_above = (window_data > threshold).mean()
    return proportion_above > proportion_threshold
```

We labeled alerts with the anomaly type in the notification, which helped technicians prioritize: point anomalies demanded immediate investigation, collective anomalies indicated trending issues, and contextual anomalies were lower priority unless the context was clearly not applicable.

---

### Q33. What features did you extract from the time series data? Did you use rolling statistics, decomposition, or frequency domain features?

We used all three categories of features, applied depending on the detection method:

**1. Rolling Statistics (Primary feature set for real-time detection):**

```python
def compute_rolling_features(series: pd.Series, windows=[15, 60, 360]) -> dict:
    features = {}
    for w in windows:  # minutes
        rolling = series.rolling(window=w)
        features[f"mean_{w}m"] = rolling.mean()
        features[f"std_{w}m"] = rolling.std()
        features[f"min_{w}m"] = rolling.min()
        features[f"max_{w}m"] = rolling.max()
        features[f"range_{w}m"] = features[f"max_{w}m"] - features[f"min_{w}m"]
        features[f"skew_{w}m"] = rolling.skew()
        features[f"pct_change_{w}m"] = series.pct_change(periods=w)
    return features
```

We computed these at three window sizes: 15 minutes (for rapid spikes), 1 hour (for short-term patterns), and 6 hours (for drift). The **rate of change** feature was particularly valuable for detecting rapid degradation.

**2. STL Decomposition (Primary method for seasonality-aware detection):**

As described in Q31, we decomposed each metric into:
- **Trend:** Long-term movement (useful for capacity planning)
- **Seasonal:** Recurring patterns (daily, weekly)
- **Residual:** What is left after removing trend and season — this is where anomalies live

We also extracted the **strength of seasonality** as a meta-feature to decide whether seasonal decomposition was even appropriate for a given asset-metric combination.

```python
def seasonality_strength(series: pd.Series, period: int) -> float:
    stl = STL(series, period=period, robust=True).fit()
    var_resid = np.var(stl.resid)
    var_deseasonalized = np.var(stl.resid + stl.seasonal)
    return max(0, 1 - var_resid / var_deseasonalized)
```

If seasonality strength was below 0.3, we skipped STL and used simpler rolling statistics.

**3. Frequency Domain Features (For periodic pattern analysis):**

We used FFT (Fast Fourier Transform) to identify dominant frequencies in the data, which helped in two ways:

- **Automatic period detection:** Rather than hardcoding seasonal periods, FFT identified the dominant periodicities. A server might have a 24-hour cycle and a 7-day cycle.
- **Spectral entropy:** High spectral entropy indicates a noisy/unpredictable signal. Low entropy indicates a regular, predictable signal. We used this to set different thresholds per asset — predictable assets got tighter anomaly thresholds.

```python
from scipy.fft import fft, fftfreq

def extract_frequency_features(series: pd.Series, sampling_rate: float) -> dict:
    N = len(series)
    yf = fft(series.values)
    xf = fftfreq(N, 1 / sampling_rate)
    
    power_spectrum = np.abs(yf[:N//2]) ** 2
    dominant_freq = xf[np.argmax(power_spectrum[1:]) + 1]  # skip DC
    
    # Spectral entropy
    ps_normalized = power_spectrum / power_spectrum.sum()
    spectral_entropy = -np.sum(ps_normalized * np.log2(ps_normalized + 1e-10))
    
    return {
        "dominant_frequency": dominant_freq,
        "dominant_period_hours": 1 / dominant_freq / 3600 if dominant_freq > 0 else None,
        "spectral_entropy": spectral_entropy
    }
```

**4. Cross-Metric Correlation Features:**

For multivariate detection (Isolation Forest), we also computed correlation features between metrics on the same asset. For example, if CPU and memory usually correlate (r=0.8) but suddenly diverge, that is anomalous even if neither metric is individually out of range.

---

### Q34. How did you handle seasonality and trend in the time series data?

Seasonality and trend handling was critical because IT infrastructure metrics have very strong periodic patterns — business hours vs nights, weekdays vs weekends, month-end processing spikes, etc.

**Seasonality Handling:**

1. **Multi-Period Seasonal Decomposition:**
Most assets exhibited at least two seasonal patterns: daily (period=24 for hourly data, 96 for 15-minute data) and weekly (period=168 for hourly data). We used MSTL (Multiple Seasonal-Trend decomposition using LOESS) to handle both:

```python
from statsmodels.tsa.seasonal import MSTL

def multi_seasonal_decompose(series: pd.Series):
    # For 15-minute interval data:
    # daily period = 96 (24h * 4), weekly period = 672 (7 * 96)
    mstl = MSTL(series, periods=[96, 672])
    result = mstl.fit()
    return result.trend, result.seasonal, result.resid
```

2. **Automatic Period Detection:**
Not all assets had the same seasonal patterns. A 24/7 server farm might have weak daily seasonality but strong weekly patterns. We used autocorrelation and FFT-based period detection to automatically identify the dominant periods per asset-metric combination:

```python
def detect_periods(series: pd.Series, max_period: int = 720) -> List[int]:
    """Detect dominant seasonal periods using autocorrelation."""
    autocorr = [series.autocorr(lag=i) for i in range(1, max_period)]
    # Find peaks in autocorrelation
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(autocorr, height=0.3, distance=10)
    return sorted(peaks + 1, key=lambda p: autocorr[p-1], reverse=True)[:3]
```

3. **Holiday and Special Event Handling:**
Regular seasonal decomposition broke down on holidays and special events (patch Tuesday, month-end closing). We maintained a calendar of known events and excluded those periods from the seasonal model training. During these events, we widened alert thresholds by a configurable factor (typically 1.5-2x).

**Trend Handling:**

1. **Trend Removal for Anomaly Detection:**
For anomaly detection, trend was removed via STL decomposition so that a gradually increasing baseline (e.g., growing database) did not cause persistent false positives.

2. **Trend Monitoring for Capacity Planning:**
The extracted trend component fed into our capacity planning alerts. We fit a linear model to the trend and projected forward:

```python
def project_threshold_breach(trend: pd.Series, threshold: float, 
                              horizon_days: int = 30) -> Optional[datetime]:
    """Predict when the trend will breach a threshold."""
    X = np.arange(len(trend)).reshape(-1, 1)
    y = trend.values
    
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    
    if slope <= 0:
        return None  # Trend is flat or decreasing
    
    current_value = y[-1]
    steps_to_breach = (threshold - current_value) / slope
    
    if steps_to_breach <= 0 or steps_to_breach > horizon_days * 96:  # 15-min intervals
        return None
    
    breach_time = trend.index[-1] + pd.Timedelta(minutes=15 * steps_to_breach)
    return breach_time
```

3. **Adaptive Baselines:**
We recalculated the baseline model weekly for each asset-metric pair. This allowed the "normal" range to adapt naturally as infrastructure usage grew. The recalculation used a trailing window of 4-8 weeks of data, with outliers removed from the training data (a chicken-and-egg problem we solved by using robust estimators).

---

### Q35. What was your alerting threshold strategy? How did you minimize false positives without missing real anomalies?

False positive reduction was our biggest operational challenge. An MSP technician receiving 50 alerts per day when only 5 are real will eventually ignore all of them — alert fatigue is dangerous.

**Threshold Strategy:**

1. **Dynamic, Per-Asset Thresholds:**
Instead of a global "CPU > 90% = alert" rule, we computed per-asset thresholds based on each asset's historical distribution. We used a percentile-based approach: an anomaly is when the value exceeds the 99.5th percentile of that asset's historical residual distribution.

```python
def compute_dynamic_threshold(historical_residuals: pd.Series, 
                                sensitivity: str = "medium") -> Tuple[float, float]:
    sensitivity_map = {
        "low": (0.5, 99.5),      # Fewer alerts, may miss subtle anomalies
        "medium": (1.0, 99.0),   # Balanced
        "high": (2.5, 97.5),     # More alerts, catches subtle anomalies
    }
    lower_pct, upper_pct = sensitivity_map[sensitivity]
    lower = np.percentile(historical_residuals, lower_pct)
    upper = np.percentile(historical_residuals, upper_pct)
    return lower, upper
```

2. **Severity-Based Multi-Threshold:**
We used three threshold levels per metric:
   - **Warning:** 97th percentile — logged but not alerted
   - **Critical:** 99th percentile — alerted with normal priority
   - **Emergency:** 99.9th percentile or absolute hard limits (e.g., disk >95%) — alerted with high priority

3. **Persistence Requirement (Debouncing):**
A single data point crossing the threshold did not trigger an alert. We required persistence — the metric had to remain anomalous for N consecutive readings (configurable, default was 3 readings = 45 minutes at 15-minute intervals for critical, 2 readings for emergency). This eliminated transient spikes:

```python
class AlertDebouncer:
    def __init__(self, required_consecutive: int = 3):
        self.consecutive_count: Dict[str, int] = defaultdict(int)
        self.required = required_consecutive
    
    def should_alert(self, asset_metric_key: str, is_anomalous: bool) -> bool:
        if is_anomalous:
            self.consecutive_count[asset_metric_key] += 1
            return self.consecutive_count[asset_metric_key] >= self.required
        else:
            self.consecutive_count[asset_metric_key] = 0
            return False
```

4. **Cross-Metric Correlation:**
We suppressed alerts where the anomaly was explained by a correlated metric. Example: if CPU spikes because of a known scheduled backup (detected by correlated disk I/O pattern), we suppressed the CPU alert and only raised a "backup running" informational notice.

5. **Alert Grouping and Deduplication:**
Multiple related anomalies (e.g., CPU, memory, and disk I/O all spiking simultaneously on the same asset) were grouped into a single alert with the root metric highlighted.

**Minimizing False Positives:**

- **Feedback Loop:** Technicians could mark alerts as "false positive" in the UI. We used this feedback to automatically widen thresholds for assets that consistently generated FPs. If an asset-metric pair got 3+ FP markings in a month, its sensitivity was automatically reduced one level.
- **Seasonal Awareness:** As described in Q34, properly accounting for seasonality eliminated the largest source of false positives (business-hours patterns misclassified as anomalies).
- **Minimum Absolute Threshold:** Even with dynamic thresholds, we enforced common-sense minimums. CPU at 45% might be high relative to a server that usually runs at 5%, but it is not operationally concerning. We required anomalies to also breach a minimum absolute impact level.

With these strategies, we achieved a false positive rate of approximately 8% (measured by technician feedback), down from 35% in the initial version.

---

### Q36. How did you handle different asset types with different baseline behaviors?

This was a significant challenge. A domain controller server has completely different "normal" behavior from a developer workstation, a print server, or a network switch.

**1. Asset Type Taxonomy:**
We defined a hierarchy of asset categories, each with default behavioral expectations:

```python
ASSET_TYPE_DEFAULTS = {
    "server_dc": {
        "cpu_baseline": "steady_moderate",  # 30-50% sustained
        "memory_baseline": "high_steady",    # 60-80% constant
        "seasonality": "weak_daily",
        "sensitivity": "high"
    },
    "server_database": {
        "cpu_baseline": "bursty",
        "memory_baseline": "high_growing",   # Grows over time
        "seasonality": "strong_daily_weekly",
        "sensitivity": "high"
    },
    "workstation": {
        "cpu_baseline": "idle_with_spikes",  # Near 0, bursts to 70%+
        "memory_baseline": "variable",
        "seasonality": "strong_daily",        # Business hours
        "sensitivity": "medium"
    },
    "network_switch": {
        "cpu_baseline": "low_steady",
        "memory_baseline": "low_steady",
        "seasonality": "strong_daily",
        "sensitivity": "medium"
    }
}
```

**2. Per-Asset Learned Profiles:**
Beyond type-level defaults, each individual asset developed its own learned profile after accumulating sufficient historical data (typically 2+ weeks). The profile captured:

```python
class AssetProfile:
    asset_id: str
    asset_type: str
    metrics: Dict[str, MetricProfile]  # Per-metric profiles

class MetricProfile:
    metric_name: str
    detected_periods: List[int]        # Auto-detected seasonal periods
    baseline_percentiles: Dict[int, float]  # p5, p25, p50, p75, p95
    spectral_entropy: float
    trend_slope: float
    seasonality_strength: float
    optimal_detection_method: str       # "stl", "ewma", "isolation_forest"
    custom_thresholds: Optional[Dict]   # Override if tuned
    last_updated: datetime
```

**3. Model Selection Per Profile:**
The detection method was chosen based on the asset's profile:

- **High seasonality (>0.5) + sufficient history:** STL/MSTL decomposition
- **Low seasonality + high noise:** EWMA with wider control limits
- **Multivariate correlations present:** Isolation Forest on the metric bundle
- **Insufficient history (<2 weeks):** Fall back to type-level defaults and static thresholds

```python
def select_detector(profile: AssetProfile, metric: str) -> AnomalyDetector:
    mp = profile.metrics[metric]
    
    if mp.seasonality_strength > 0.5 and len(mp.detected_periods) > 0:
        return STLDetector(periods=mp.detected_periods)
    elif mp.spectral_entropy > 4.0:  # High noise
        return EWMADetector(span=20, control_sigma=3.0)
    else:
        return ZScoreDetector(window=360, threshold=3.5)
```

**4. Profile Clustering:**
For new assets or assets with sparse data, we used profile clustering. We embedded existing asset profiles into a feature vector (baseline percentiles, seasonality metrics, etc.) and used K-means clustering to group similar assets. A new asset was assigned to the nearest cluster, and that cluster's aggregate thresholds were used until the asset built enough individual history.

**5. Customer-Level Overrides:**
MSPs could configure custom thresholds per asset or asset group through the SuperOps UI. These overrides took precedence over learned profiles. This was essential because some customers had unique operational patterns that no amount of automated learning would capture (e.g., "this server always spikes at 2 AM because of a legacy batch job, ignore it").

---

### Q37. What was the data pipeline architecture — how did data flow from assets to your anomaly detection service?

The data pipeline was a multi-stage streaming architecture designed for both real-time alerting and batch model training:

**Architecture Overview:**

```
[RMM Agents on Assets]
        |
        v  (HTTPS/gRPC, 1-5 min intervals)
[Ingestion API (FastAPI)]
        |
        v
[Apache Kafka] ---- topic: raw_metrics
        |
        +--> [Stream Processor (Faust/Kafka Streams)]
        |         |
        |         +--> Real-time anomaly detection
        |         +--> Publish to topic: alerts
        |         +--> Publish to topic: enriched_metrics
        |
        +--> [Batch Consumer]
                  |
                  +--> Write to TimescaleDB (time series storage)
                  +--> Weekly model retraining job
        
[TimescaleDB] <----> [Model Training Service (Celery)]
                            |
                            v
                    [Model Registry (Redis/S3)]
                            |
                            v
                    [Stream Processor reads latest models]
```

**Detailed Flow:**

**Stage 1 - Ingestion:**
RMM agents installed on customer assets collected metrics (CPU, memory, disk, network, process-level stats) at configurable intervals (default 5 minutes). Agents batched metrics and sent them via HTTPS to our Ingestion API. The API performed basic validation (schema check, timestamp sanity, deduplication) and published to Kafka.

```python
@app.post("/v1/metrics")
async def ingest_metrics(payload: MetricsBatch):
    validated = validate_and_normalize(payload)
    for metric in validated:
        await kafka_producer.send(
            topic="raw_metrics",
            key=f"{metric.asset_id}:{metric.metric_name}",
            value=metric.to_json()
        )
    return {"status": "accepted", "count": len(validated)}
```

**Stage 2 - Stream Processing:**
A Faust-based stream processor consumed from `raw_metrics`. For each incoming data point, it:
1. Loaded the asset's current profile and model from Redis cache
2. Computed rolling features incrementally (maintained state in RocksDB via Faust tables)
3. Ran the appropriate anomaly detector
4. Applied debouncing logic
5. If anomaly confirmed, published to `alerts` topic

```python
@app.agent(raw_metrics_topic)
async def process_metric(stream):
    async for event in stream:
        profile = await profile_cache.get(event.asset_id)
        detector = select_detector(profile, event.metric_name)
        
        # Update rolling state
        state = rolling_state[event.asset_id][event.metric_name]
        state.update(event.value, event.timestamp)
        
        # Detect
        is_anomalous, score, details = detector.detect(
            value=event.value,
            rolling_state=state,
            profile=profile
        )
        
        if debouncer.should_alert(event.asset_key, is_anomalous):
            await alerts_topic.send(Alert(
                asset_id=event.asset_id,
                metric=event.metric_name,
                severity=compute_severity(score),
                details=details
            ))
```

**Stage 3 - Storage and Batch Processing:**
A separate consumer wrote enriched metrics to TimescaleDB with continuous aggregation policies (automatic rollups from raw -> 1-hour -> 1-day granularity). This kept storage costs manageable across tens of thousands of assets.

A weekly Celery job retrained the seasonal models and updated asset profiles using the latest 4-8 weeks of data from TimescaleDB. Updated models were stored in Redis (for lightweight models like thresholds) or S3 (for Isolation Forest model files), and the stream processor picked them up on the next polling cycle.

**Stage 4 - Alert Delivery:**
The `alerts` topic was consumed by the Alert Routing Service, which applied grouping, deduplication, and routing logic (email, Slack, PagerDuty, in-app notification) based on customer preferences and severity.

---

### Q38. How did you evaluate the anomaly detection system's performance? What metrics (precision, recall, F1) did you achieve?

**Evaluation Methodology:**

Evaluating anomaly detection is inherently challenging because ground truth labels are sparse and subjective. We used three evaluation approaches:

**1. Labeled Test Set (Point-in-Time Evaluation):**
We curated a labeled dataset of ~2,500 time series windows (each 1 week long) across different asset types and metrics. Labels came from two sources:
- **Known incidents:** We backfilled labels from our incident management system — if a ticket was filed about a server issue at time T, the corresponding metric anomaly window was labeled as "true anomaly."
- **Expert annotation:** Senior engineers manually reviewed and labeled a random sample of 500 time series windows, marking anomalous regions.

Each annotated anomaly was categorized (point, contextual, collective) and rated by severity (minor, major, critical).

**2. Production Feedback Loop (Ongoing Evaluation):**
Technicians could mark alerts as:
- True Positive: Real issue, alert was helpful
- False Positive: Not a real issue
- True Negative (implicit): No alert, no issue
- False Negative: Discovered via customer complaint — should have alerted

**3. Synthetic Injection (Stress Testing):**
We injected synthetic anomalies into historical data (spike injection, trend injection, level shift injection) and measured detection rates.

**Results:**

| Metric | Labeled Test Set | Production (30-day rolling) |
|--------|-----------------|----------------------------|
| Precision | 91.2% | 88.5% |
| Recall | 87.4% | 83.1% |
| F1 Score | 89.3% | 85.7% |
| False Positive Rate | 8.8% | 11.5% |
| Mean Detection Latency | 12 min | 18 min |

**Breakdown by anomaly type:**

| Type | Precision | Recall |
|------|-----------|--------|
| Point anomalies | 94% | 92% |
| Contextual anomalies | 89% | 85% |
| Collective anomalies | 86% | 78% |

Collective anomalies had the lowest recall because gradual drift is inherently harder to detect, and the boundary between "normal growth" and "anomalous drift" is fuzzy.

**Key Observations:**
- Production metrics were lower than test set metrics due to concept drift and the broader diversity of real-world scenarios.
- The biggest precision gains came from debouncing and seasonal adjustment (added ~15% precision each).
- The biggest recall gains came from multi-period seasonal decomposition (previously, weekly patterns were being missed).
- We tracked these metrics on a weekly dashboard and set alert-on-alerts: if the rolling 7-day false positive rate exceeded 15%, the on-call engineer was notified to investigate.

---

### Q39. How did you handle cold-start — when a new asset has no historical data?

Cold-start was a real operational concern since MSPs onboard new assets regularly. We used a progressive approach:

**Phase 1 (Day 0 - No data): Type-Based Defaults**

When an asset was first enrolled, we applied conservative static thresholds based on asset type and operating system:

```python
COLD_START_THRESHOLDS = {
    ("server", "windows"): {
        "cpu_percent": {"warning": 85, "critical": 95},
        "memory_percent": {"warning": 85, "critical": 95},
        "disk_percent": {"warning": 80, "critical": 90},
    },
    ("workstation", "windows"): {
        "cpu_percent": {"warning": 90, "critical": 98},
        "memory_percent": {"warning": 90, "critical": 95},
        "disk_percent": {"warning": 85, "critical": 95},
    },
    # ... more asset types
}
```

These were intentionally conservative (wider thresholds) to avoid flooding new customers with false positives during onboarding.

**Phase 2 (Days 1-7 - Learning period): Cluster Assignment**

After 24 hours of data, we computed basic statistical features (mean, std, percentiles) and assigned the asset to the nearest behavioral cluster from our existing asset population. This cluster's aggregate thresholds provided better-than-default thresholds without needing asset-specific history.

```python
def assign_to_cluster(new_asset_features: np.ndarray, 
                       cluster_model: KMeans) -> int:
    cluster_id = cluster_model.predict(new_asset_features.reshape(1, -1))[0]
    return cluster_id

def get_cluster_thresholds(cluster_id: int) -> Dict:
    # Return the aggregate thresholds computed from all assets in this cluster
    return cluster_threshold_store[cluster_id]
```

**Phase 3 (Days 7-14 - Preliminary profiling): Partial Profile**

After 1 week, we could detect the daily seasonal pattern (if present) and compute rolling statistics. We switched from cluster thresholds to asset-specific thresholds for metrics that showed clear patterns, while keeping cluster thresholds for metrics that were still noisy.

**Phase 4 (Days 14+ - Full profiling): Individual Profile**

After 2+ weeks, the asset had a full individual profile with multi-period seasonal decomposition. The system transitioned to the standard detection pipeline.

**Handling the Transition:**

The transitions between phases were not hard cutoffs. We used a blending approach:

```python
def blended_threshold(asset_age_days: int, 
                       individual_threshold: float,
                       cluster_threshold: float) -> float:
    # Gradually shift from cluster to individual over 14 days
    individual_weight = min(1.0, asset_age_days / 14.0)
    cluster_weight = 1.0 - individual_weight
    return (individual_weight * individual_threshold + 
            cluster_weight * cluster_threshold)
```

This prevented sudden threshold jumps as the system transitioned between phases.

**UI Transparency:**
We displayed the asset's current profiling phase in the UI with a "learning" badge and a progress indicator. This set expectations with MSP technicians that alerts might be less accurate for newly onboarded assets.

---

### Q40. Did you consider using autoencoders or LSTMs for anomaly detection? Why or why not?

Yes, we evaluated both. Here is the analysis:

**Autoencoder Evaluation:**

We prototyped a vanilla autoencoder and a variational autoencoder (VAE) that took a window of 96 data points (24 hours at 15-minute intervals) as input and tried to reconstruct it. Anomalies were flagged when reconstruction error exceeded a threshold.

*Pros we observed:*
- Good at detecting collective anomalies (unusual patterns over a window)
- Could handle multivariate input naturally (feed CPU+memory+disk as a single input)
- No need to explicitly model seasonality — the autoencoder learned it

*Cons that stopped us from production deployment:*
- **Per-asset training required:** One global autoencoder across all assets performed poorly because different assets have different "normal" patterns. Training per-asset was feasible but meant managing tens of thousands of models with regular retraining.
- **Threshold tuning was just as hard:** We still needed to set a reconstruction error threshold per asset. This was conceptually similar to setting a Z-score threshold but less interpretable.
- **Explainability gap:** When the autoencoder flagged an anomaly, we could not easily explain which aspect of the time window was abnormal. With STL, we could say "the residual after removing daily seasonality spiked" which is actionable.
- **Training instability:** Some assets with highly irregular data produced autoencoders with unstable reconstruction error distributions, leading to erratic alerting.

**LSTM Evaluation:**

We prototyped an LSTM that predicted the next N data points and flagged anomalies when the actual value deviated significantly from the prediction.

*Pros:*
- Naturally captured temporal dependencies
- Slightly better recall on collective anomalies (89% vs 78% with our statistical approach)
- Could handle irregular sampling intervals better

*Cons:*
- **Latency:** LSTM inference on CPU was 50-100ms per prediction vs <1ms for statistical methods. At scale (50,000+ time series), this added up significantly.
- **Training data requirements:** Needed 4-8 weeks of data minimum for stable training per asset (vs 2 weeks for statistical methods). This worsened the cold-start problem.
- **GPU dependency for training:** Retraining tens of thousands of LSTM models weekly required GPU resources we did not want to manage.
- **Marginal accuracy gain:** Overall F1 improved from 89.3% to 91.1% — a 2% improvement that did not justify the operational complexity.

**Our Conclusion:**

We followed a pragmatic engineering principle: if a simpler model gets you 90% of the way there, the complexity cost of the remaining 10% must be justified by clear business value. In our case, the 2% F1 improvement from deep learning did not justify:
- 10x increase in infrastructure cost (GPU compute)
- 5x increase in operational complexity (model training pipelines, model versioning, GPU management)
- Worse cold-start behavior
- Loss of interpretability

However, we did identify a specific use case where deep learning might be worth revisiting: multivariate collective anomaly detection on high-value server clusters where the extra recall on gradual degradation patterns could prevent expensive outages. We flagged this as a future exploration.

---

## Part 3: Ticket Classification, Routing, and Similar Ticket Detection (Q41-Q45)

---

### Q41. How did you implement ticket classification — what model/approach did you use for auto-categorization by priority?

We implemented a multi-label classification system that assigned tickets three properties: **category** (network, hardware, software, account, security), **sub-category** (more specific), and **priority** (P1-P4). The approach evolved through three iterations:

**Iteration 1 - Rule-Based Baseline:**
We started with keyword-based rules as a baseline: tickets containing "server down," "outage," or "cannot access" were classified as P1. This achieved ~55% accuracy overall but was brittle and required constant rule maintenance.

**Iteration 2 - Classical ML (Production v1):**
We trained a pipeline of text classification models:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

# Preprocessing
def preprocess_ticket(ticket: dict) -> str:
    """Combine subject, body, and metadata into a single text."""
    parts = [
        ticket.get("subject", ""),
        ticket.get("body", ""),
        f"channel:{ticket.get('channel', 'unknown')}",
        f"asset_type:{ticket.get('asset_type', 'unknown')}",
        f"customer_tier:{ticket.get('customer_tier', 'unknown')}"
    ]
    return " ".join(parts)

# Model pipeline
category_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000, 
        ngram_range=(1, 3),
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        C=1.0, 
        class_weight="balanced",  # Handle class imbalance
        max_iter=1000
    ))
])

priority_model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 3))),
    ("classifier", LogisticRegression(C=1.0, class_weight="balanced"))
])
```

This achieved ~78% accuracy on category and ~72% on priority.

**Iteration 3 - Embedding-Based with Fine-Tuned Transformer (Production v2):**
We fine-tuned a DistilBERT model on our ticket corpus:

```python
from transformers import DistilBertForSequenceClassification, Trainer

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(category_labels),
    problem_type="single_label_classification"
)

# Training with our labeled ticket data (~15,000 labeled tickets)
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./ticket_classifier",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    ),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
```

This achieved ~87% on category and ~82% on priority.

**Priority Classification Specifics:**

Priority was the hardest to classify because it depends on context beyond the text:
- Customer SLA tier (enterprise customers get auto-upgraded)
- Business hours vs off-hours
- Number of affected users
- Asset criticality

We used a two-stage approach: the DistilBERT model predicted a base priority from text, then a rule layer adjusted it based on structured metadata:

```python
def final_priority(text_priority: str, metadata: dict) -> str:
    priority = text_priority  # P1, P2, P3, P4
    
    # Upgrade rules
    if metadata["customer_tier"] == "enterprise" and priority in ["P3", "P4"]:
        priority = upgrade_one_level(priority)
    if metadata["affected_users"] > 10:
        priority = upgrade_one_level(priority)
    if metadata["asset_criticality"] == "business_critical":
        priority = upgrade_one_level(priority)
    
    # Hard overrides
    if any(kw in metadata["text_lower"] for kw in ["security breach", "data loss"]):
        priority = "P1"
    
    return priority
```

**Deployment:**
The classifier ran as a FastAPI microservice. The DistilBERT model was served via ONNX Runtime for faster inference (~15ms per ticket). We used model versioning with MLflow and canary deployments for model updates.

---

### Q42. How did the similar ticket detection engine work? Did you use embeddings, TF-IDF, or something else?

We used a hybrid approach combining dense embeddings for semantic similarity with sparse features for keyword precision:

**Architecture:**

```
[New Ticket]
     |
     +--> [Embedding Model] --> dense vector (768d)
     |
     +--> [TF-IDF] --> sparse vector
     |
     v
[Hybrid Search: weighted combination]
     |
     v
[Candidate Retrieval from Vector DB (top 50)]
     |
     v
[Re-Ranking Model (cross-encoder)]
     |
     v
[Top 5 Similar Tickets with Resolutions]
```

**Stage 1 - Embedding Generation:**
We used a sentence-transformer model (`all-MiniLM-L6-v2` initially, later fine-tuned on our ticket data) to generate 384-dimensional dense embeddings for each ticket.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_ticket(ticket: dict) -> np.ndarray:
    text = f"{ticket['subject']} {ticket['body']}"
    # Truncate to 512 tokens
    return model.encode(text, normalize_embeddings=True)
```

**Stage 2 - Indexing:**
All historical resolved tickets were indexed in a vector database (we used Qdrant). Each ticket was stored with its embedding, metadata (category, priority, resolution time), and the resolution text.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(host="qdrant-service", port=6333)

client.create_collection(
    collection_name="resolved_tickets",
    vectors_config={
        "dense": VectorParams(size=384, distance=Distance.COSINE),
    }
)

# Upsert a resolved ticket
client.upsert(
    collection_name="resolved_tickets",
    points=[{
        "id": ticket_id,
        "vector": {"dense": embedding.tolist()},
        "payload": {
            "subject": ticket["subject"],
            "body": ticket["body"][:500],
            "resolution": ticket["resolution"],
            "category": ticket["category"],
            "resolved_date": ticket["resolved_date"],
            "customer_id": ticket["customer_id"]
        }
    }]
)
```

**Stage 3 - Hybrid Retrieval:**
For a new ticket, we performed both dense (embedding) and sparse (keyword) search and combined the results:

```python
async def find_similar_tickets(ticket: dict, top_k: int = 50) -> List[dict]:
    embedding = embed_ticket(ticket)
    
    # Dense search
    dense_results = client.search(
        collection_name="resolved_tickets",
        query_vector=("dense", embedding),
        limit=top_k,
        query_filter=Filter(
            must=[
                FieldCondition(key="category", match=MatchValue(value=ticket["category"]))
            ]
        )
    )
    
    # Sparse keyword search (BM25 via Elasticsearch sidecar)
    sparse_results = es_client.search(
        index="resolved_tickets",
        body={"query": {"match": {"body": ticket["body"]}}},
        size=top_k
    )
    
    # Reciprocal Rank Fusion to combine results
    combined = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    return combined[:top_k]
```

**Stage 4 - Re-Ranking:**
The top 50 candidates were re-ranked using a cross-encoder model that took the (query ticket, candidate ticket) pair and produced a more accurate similarity score:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query_ticket: dict, candidates: List[dict]) -> List[dict]:
    pairs = [
        (f"{query_ticket['subject']} {query_ticket['body']}", 
         f"{c['subject']} {c['body']}")
        for c in candidates
    ]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked[:5] if s > 0.3]  # Minimum relevance threshold
```

**Why Hybrid?**

Pure embedding search was great for semantic similarity ("computer won't turn on" matches "workstation not booting") but missed exact error codes and technical terms. Pure TF-IDF caught exact matches but missed semantic paraphrases. The hybrid approach gave us the best of both worlds, achieving ~85% relevance in our evaluations (measured as "at least one of top-3 results is genuinely helpful for resolving the ticket").

---

### Q43. What similarity metric did you use (cosine similarity, Euclidean, etc.)? What was the retrieval architecture?

**Similarity Metric:**

We used **cosine similarity** as our primary metric for dense embeddings, and here is why:

1. **Scale invariance:** Cosine similarity measures the angle between vectors, not their magnitude. This is important because embedding magnitude can vary based on text length — a short "server down" ticket and a long detailed description of the same problem should still be considered similar.

2. **Normalized efficiency:** We pre-normalized all embeddings to unit length at indexing time, which meant cosine similarity reduced to a simple dot product — the fastest vector operation available in vector databases.

3. **Empirical superiority:** We benchmarked cosine vs Euclidean vs dot product on our labeled evaluation set:
   - Cosine: Recall@5 = 78.3%, MRR = 0.72
   - Euclidean (L2): Recall@5 = 71.5%, MRR = 0.66
   - Dot Product (unnormalized): Recall@5 = 75.1%, MRR = 0.69

For the sparse (BM25) component, we used Okapi BM25 scoring via Elasticsearch, which is the standard for keyword relevance and accounts for term frequency, document length normalization, and inverse document frequency.

**Retrieval Architecture in Detail:**

```
                    ┌──────────────────────────┐
                    │    FastAPI Service        │
                    │   /api/similar-tickets    │
                    └───────────┬──────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │   Query Preprocessor      │
                    │   - Clean text            │
                    │   - Extract entities      │
                    │   - Generate embedding    │
                    └───────────┬──────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
    ┌───────────▼──────┐  ┌────▼────────┐  ┌───▼──────────┐
    │  Qdrant (Dense)  │  │ Elasticsearch│  │ Metadata     │
    │  ANN Search      │  │ BM25 Search  │  │ Filters      │
    │  Top 50          │  │ Top 50       │  │ (category,   │
    │                  │  │              │  │  date range)  │
    └───────────┬──────┘  └────┬────────┘  └───┬──────────┘
                │               │               │
                └───────────────┼───────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │ Reciprocal Rank Fusion    │
                    │ Merge & deduplicate       │
                    │ Top 50 combined           │
                    └───────────┬──────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │ Cross-Encoder Re-Ranker   │
                    │ Score each pair           │
                    │ Top 5 with scores > 0.3   │
                    └───────────┬──────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │ Response Formatter         │
                    │ - Ticket summaries         │
                    │ - Resolution excerpts      │
                    │ - Confidence scores        │
                    └──────────────────────────┘
```

**Indexing Architecture:**

The index was kept up-to-date via an event-driven pipeline:

```python
# When a ticket is resolved, it gets indexed
@event_handler("ticket.resolved")
async def index_resolved_ticket(event: TicketResolvedEvent):
    ticket = await ticket_service.get(event.ticket_id)
    
    # Only index tickets with meaningful resolutions
    if len(ticket.resolution) < 20:
        return  # Skip tickets closed without resolution
    
    embedding = embed_ticket(ticket)
    
    # Upsert to Qdrant
    await qdrant_client.upsert(
        collection_name="resolved_tickets",
        points=[PointStruct(
            id=ticket.id,
            vector={"dense": embedding.tolist()},
            payload={
                "subject": ticket.subject,
                "body": ticket.body[:500],
                "resolution": ticket.resolution,
                "category": ticket.category,
                "priority": ticket.priority,
                "resolved_date": ticket.resolved_date.isoformat(),
                "resolution_time_hours": ticket.resolution_time_hours,
                "customer_id": ticket.customer_id
            }
        )]
    )
    
    # Also index in Elasticsearch for BM25
    await es_client.index(
        index="resolved_tickets",
        id=ticket.id,
        body={...}
    )
```

**Scale Considerations:**

With ~500K resolved tickets in the index, Qdrant's HNSW index provided sub-10ms search latency for the dense component. Elasticsearch BM25 was similarly fast. The cross-encoder re-ranking was the bottleneck at ~50ms for 50 candidates, but this was acceptable for our SLA.

We set up a nightly compaction job that removed tickets older than 2 years (unless they were highly cited) to keep the index fresh and relevant.

---

### Q44. How did you handle the cold-start problem when the system had few historical tickets?

This was a real challenge during initial deployment and when onboarding new MSP customers who had no ticket history in SuperOps.

**Strategy 1 - Pre-Seeded Knowledge Base:**

Before any customer-specific tickets existed, we seeded the system with:
- ~5,000 curated Q&A pairs from public IT support forums (Stack Overflow Server Fault, Spiceworks community)
- ~2,000 entries from our own product documentation and KB articles
- ~500 synthetic tickets generated by our team, covering common MSP scenarios

These pre-seeded entries were tagged as "generic" and ranked lower than customer-specific tickets once those became available.

```python
def retrieval_score(candidate: dict, query: dict) -> float:
    base_score = candidate["similarity_score"]
    
    # Boost customer-specific tickets
    if candidate["source"] == "customer_tickets":
        base_score *= 1.3
    elif candidate["source"] == "generic_kb":
        base_score *= 1.0
    elif candidate["source"] == "seeded_community":
        base_score *= 0.8  # Slight penalty — less reliable
    
    # Recency boost
    days_old = (now - candidate["resolved_date"]).days
    recency_factor = max(0.7, 1.0 - days_old / 730)  # Decay over 2 years
    base_score *= recency_factor
    
    return base_score
```

**Strategy 2 - Cross-Customer Learning (with Privacy Controls):**

With customer consent (configured per MSP organization), we enabled anonymized cross-customer ticket matching. A new customer with zero tickets could benefit from the 500K tickets of other customers. We anonymized by:
- Stripping customer names, IP addresses, and identifiers
- Replacing specific hostnames with generic labels
- Only sharing the problem description and resolution steps, not internal notes

```python
# Cross-customer search filter
def build_search_filter(customer_id: str, cross_customer_enabled: bool):
    if cross_customer_enabled:
        return Filter(
            should=[
                FieldCondition(key="customer_id", match=MatchValue(value=customer_id)),
                FieldCondition(key="anonymized", match=MatchValue(value=True))
            ]
        )
    else:
        return Filter(
            must=[
                FieldCondition(key="customer_id", match=MatchValue(value=customer_id))
            ]
        )
```

**Strategy 3 - Active Learning During Ramp-Up:**

During the first 30 days of deployment, we ran the system in "shadow mode" — it generated classifications and similar ticket suggestions but did not auto-act on them. Human agents saw the AI suggestions as recommendations and could accept or reject them. Every accept/reject was used as a training signal to rapidly adapt the models to the customer's specific ticket patterns.

```python
class ActiveLearningCollector:
    async def record_feedback(self, ticket_id: str, suggestion_id: str, 
                               accepted: bool, corrected_label: Optional[str]):
        await self.feedback_store.save({
            "ticket_id": ticket_id,
            "suggestion_id": suggestion_id,
            "accepted": accepted,
            "corrected_label": corrected_label,
            "timestamp": datetime.utcnow()
        })
        
        # Trigger model fine-tuning after accumulating enough feedback
        feedback_count = await self.feedback_store.count_since_last_training()
        if feedback_count >= 100:
            await self.training_queue.enqueue("incremental_finetune")
```

**Strategy 4 - LLM-Based Classification as Fallback:**

When ML models had low confidence (below 0.6 probability for the top class), we fell back to an LLM-based classification using few-shot prompting. The LLM was given the category taxonomy and 3-5 example tickets per category, and asked to classify the new ticket. This was more expensive and slower but more robust in low-data regimes.

The cold-start period typically lasted 2-4 weeks for classification (until we had ~200-300 labeled tickets) and 4-6 weeks for similar ticket detection (until the index had ~500+ resolved tickets). After that, the system's accuracy matched our benchmarks.

---

### Q45. How did you evaluate the ticket triaging accuracy? What was the baseline vs your system?

**Evaluation Framework:**

We evaluated three aspects of triaging: classification accuracy (correct category/priority), routing accuracy (assigned to the right team/person), and business impact (time saved, SLA compliance).

**Baselines:**

1. **Manual baseline:** Human agents classifying and routing tickets. We measured this by tracking reclassification and rerouting rates — how often a ticket was initially classified/routed incorrectly and had to be changed. The manual error rate was ~22% for category and ~30% for priority.

2. **Rule-based baseline:** The keyword-based system that preceded our ML approach. This achieved ~55% category accuracy and ~48% priority accuracy.

**Our System's Performance:**

| Metric | Manual | Rule-Based | ML System (v1 - TF-IDF) | ML System (v2 - DistilBERT) |
|--------|--------|------------|--------------------------|------------------------------|
| Category Accuracy | 78% | 55% | 78% | 87% |
| Priority Accuracy | 70% | 48% | 72% | 82% |
| Routing Accuracy | 72% | 45% | 68% | 79% |
| Reclassification Rate | 22% | 45% | 22% | 13% |
| Rerouting Rate | 28% | 55% | 32% | 21% |

**Evaluation Methodology:**

1. **Offline Evaluation (Test Set):**
We maintained a held-out test set of 3,000 tickets with human-verified labels, refreshed quarterly. Standard classification metrics were computed:

```python
from sklearn.metrics import classification_report, confusion_matrix

# Category classification
print(classification_report(y_true_category, y_pred_category))

# Priority classification
print(classification_report(y_true_priority, y_pred_priority))

# Confusion matrix to identify systematic misclassifications
cm = confusion_matrix(y_true_priority, y_pred_priority, labels=["P1","P2","P3","P4"])
```

Key findings from confusion matrix analysis:
- The biggest confusion was between P2 and P3 — the boundary is genuinely ambiguous even for humans.
- P1 precision was 94% (critical — we rarely cry wolf on P1) but P1 recall was 88% (we missed some genuine P1s that were described in understated language).

2. **Online A/B Evaluation:**
We ran the ML system in parallel with human classification for 4 weeks. Tickets were classified by both, and a senior agent adjudicated disagreements. This gave us head-to-head accuracy comparisons on the exact same ticket population.

3. **Business Impact Metrics:**

| Metric | Before AI | After AI | Improvement |
|--------|-----------|----------|-------------|
| Mean Time to Triage (MTTT) | 15 min | <1 sec | 99.9% |
| Mean Time to First Response | 45 min | 8 min* | 82% |
| SLA Breach Rate | 12% | 5% | 58% |
| Agent Throughput (tickets/day) | 35 | 52 | 49% |

*8 minutes includes agent review time for AI-classified tickets

4. **Confidence-Stratified Analysis:**
We analyzed accuracy by model confidence level:

| Confidence Band | % of Tickets | Accuracy |
|----------------|-------------|----------|
| High (>0.9) | 45% | 95% |
| Medium (0.7-0.9) | 35% | 82% |
| Low (<0.7) | 20% | 65% |

This informed our auto-classification policy: tickets with high confidence were auto-triaged without human review, medium confidence tickets showed the AI suggestion but required human confirmation, and low confidence tickets were manually triaged with the AI suggestion as a hint.

5. **Continuous Monitoring:**

We tracked weekly accuracy metrics and set up drift detection alerts. If classification accuracy dropped below 80% on a rolling 7-day window (measured via agent corrections), we triggered an investigation. Common causes of drift included:
- New product features generating ticket types the model had not seen
- Seasonal shifts (e.g., tax season for accounting software MSPs)
- Changes in the customer base composition

We retrained the model monthly on the accumulated new data, with an automated pipeline that would retrain earlier if drift was detected.

---

*Prepared for Deepaksakthi - SuperOps AI/LLM Engineering Interview Prep*
