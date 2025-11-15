# `untitled` Agentic RL Design Doc
@kzhao @zhaoranwang

## Motivating Principles
We prioritize simplicity and modularity for the first design of this project. I aim to write in a way to best avoid type ambiguity. The end goal is for the user to be able to interface with our library like the following:

### What are the core concepts we need to solve and support (ordered by priority)?
- Rollout, which should be done asynchronously from training
- Model versioning and tracking (this is probably one of the most important things)
- A trajectory queue for ingestion of rollouts (meant for training)
- Training module, supporting a variety of different algorithms: PPO, GRPO, etc
- Context management tools (RAG, memory, vector DBs, context compaction, etc)

We DO NOT optimize for ease of use in this initial design. User will need to do fundamental state transition management and action generation / computation. We can however later write abstractions such that these are all taken care under the hood. If our goal is to create a `pytorch` then what this initial design should be is like `numpy`. 

Since the user needs to do manual state transition management and action generation / computation, the functionality of this document is similar to that of `agent-lightning`.

### Core interfaces
#### Context and message management (nothing complicated just yet no RAG, etc)

- `Message`: A singular item in the conversation history. The user can optionally implement a renderer, which translates a list of `Message` into a chat-templated string for actual LLM consumption. Default ones will be provided and used for the most common cases.
```
class Message(BaseModel):
  role: Literal["system", "user", "assistant", "tool"]
  content: str

Example of message list:
[
  {"role": "system", "content": "...}
  {"role": "user", "content": "...}
  {"role": "assistant", "content": "...}
  {"role": "tool", "content": "...}
]
```

- `Renderer`: A renderer translates a list of `Message` into a chat-templated string for actual LLM consumption. But the `Renderer` can be more complicated and can strip messages of their `<think></think>` tags if the user implements. This for example is recommended by Qwen3 models.
```text
{"role": "system", "content": "You are a helpful assistant."}
{"role": "user", "content": "What is the capital of France?"}
{"role": "assistant", "content": "Paris"}

||
||
||
vv

<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is the capital of France?
<|im_end|>
<|im_start|>assistant
Paris
<|im_end|>
```

- `Tokenizer`: These are just going to be taken straight from HuggingFace or whatever API these models use for tokenization. An issue here is tokenizer instability (which can experimentally verify)

Beyond these, the user can implement any context management tools they want. Our API is generic enough to not care about the implementation details.

#### Rollout management abstractions
- `TrajectoryQueue`: A trajectory is essentially a chronologically ordered list of $(s, a, r, s')$ tuples with critic reward $r$ computed by the user. 

```python
class TrajectoryQueue:
  @classmethod
  def push(
    trajectory_id: str,
    timestep: int,
    state: str,
    action: str,
    reward: float,
    # Which model generated the action (like the model id)
    # The tokenizer necessary to encode the action
    # Other necessary metadata to combat trajectory staleness
    metadata: dict,
  ):
    ...

  @classmethod
  def pop(trajectory_id: str) -> Trajectory:
    ...

  @classmethod
  def register(trajectory_id: str) -> Trajectory:
    ...

  @classmethod
  def complete(trajectory_id: str) -> Trajectory:
    ...

  @classmethod
  def get_trajectory(trajectory_id: str) -> Trajectory:
    ...
```

#### Model interface
- `ModelManager`: Something that keeps track of model versioning and loading
```python
class Model:
  model_id: str
  base_model: str # eg. "Qwen/Qwen2.5-7B-Instruct"

  def checkpoint(is_sampling_mode: bool):
    ...

class ModelManager:
  @classmethod
  def register(model_id: str, model_name: str): 
    ...

  @classmethod
  def load(model_id: str) -> Model:
    ...
```
#### TransitionManager
- `TransitionManager`: We want the user to write as few lines of boilerplate code as possible, especially in transition management, so we ask them to implement one of these per task.
```python
class TransitionManager:
  @classmethod
  def transition(
    curr_state: str,
    action: str,
    reward_func: Callable,
  ) -> NextState, Reward, Done, Info:
    pass
```
- This is the probably the most loaded part of the entire document, since it encapsulates so much. In the (supervised target case), for example Hendryck's math, we have target responses, this transition manager will note which example we are on and generate state accordingly. In live generation case, it is much simpler, where we have a raw reward_callable  

#### Training interface
```python
class Optimizer:
  algorithm: Literal["ppo", "grpo"]
  trajectory_ingest_strategy: Literal[
    "greedy",
    "on-policy",
    "off-policy-with-staleness-control",
    ...
  ]
  ...

optimizer = Optimizer(...)
optimizer.fit(rollout_func, ...)
```

## Frontend
This is how I imagine the frontend to be used by the user. I will provide an example where the model has to guess a number that the user is thinking of between 1 and 100. We restrict to the scenario where the model cannot change DURING the trajectory (for now).

```python
@rollout_manager.register("example_rollout")
def example_rollout(model_id: str, trajectory_id: str):
  model = ModelManager.load(model_id)
  TrajectoryQueue.register(trajectory_id)

  user_number = random.randint(1, 100)
  current_state = [{ "role": "user", "content": "Guess my number between 1 and 100." }]
  while True:
    completion = model.complete(current_state)
    action = extract_action(completion)
    next_info = TransitionManager.transition(
      current_state,
      action,
      reward_func,
    )
    if next_info.done:
      TrajectoryQueue.complete(trajectory_id)
      return
    current_state = next_info.state
    TrajectoryQueue.push(
      trajectory_id,
      current_state,
      action,
      next_info.reward,
      metadata,
    )

optimizer = Optimizer(...)
optimizer.fit(example_rollout)
```

#### Nice to haves
- `ContextCompresssion`: Take a list of `Message` and compress the conversation history
- Arbitrary context management tools
- `Tool` registry and execution
  - Enables memory and RAG
- Agent Teams / Multi-agent systems
  - With the above API, we can support arbitrary number of agents, as long as there are "handoffs" implemented between rollouts   
  - Voting
  - Cooperative / Competitive
- Human in the loop RL (for critical applications where the application will stop and ask the user to reward the model)
