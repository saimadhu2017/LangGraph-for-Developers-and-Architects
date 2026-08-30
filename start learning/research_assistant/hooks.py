from .state import ResearchState


def pre_summarize(state: ResearchState) -> ResearchState:
    print(f"\n[PRE-HOOK]  question : '{state['question']}'")
    print(f"[PRE-HOOK]  keywords : {state.get('keywords', [])}")
    print(f"[PRE-HOOK]  sources  : {len(state.get('sources', []))} doc(s)")
    return state


def post_summarize(state: ResearchState) -> ResearchState:
    summary = state.get("summary", "")
    print(f"[POST-HOOK] summary  : {len(summary)} chars generated")
    return state


def wrap_with_hooks(node_fn, pre=None, post=None):
    """Returns a new node function that calls pre → node_fn → post.

    The pre-hook's return is used as the node's input (so it can modify state).
    The post-hook receives the merged state for observation; its return is ignored
    in this pattern, so use it for logging and side effects only.
    """
    def wrapped(state: ResearchState) -> dict:
        if pre:
            state = pre(state)
        result = node_fn(state)
        if post:
            post({**state, **result})
        return result
    return wrapped
