
try:
    from pydantic_ai.ui.ag_ui import AGUIAdapter
    import inspect
    
    print(inspect.getsource(AGUIAdapter.run_stream))
        
except ImportError:
    print("Could not import AGUIAdapter")
except Exception as e:
    print(f"Error: {e}")
