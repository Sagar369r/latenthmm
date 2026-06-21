import json
import os

class StateSaver:
    """
    Saves and loads Kalman, CUSUM, and other stateful filters 
    to disk every 60 seconds to recover gracefully from crashes.
    """
    def __init__(self, save_dir: str = "models/state", interval_sec: float = 60.0):
        self.save_dir = save_dir
        self.interval_sec = interval_sec
        self.last_save = 0.0
        os.makedirs(self.save_dir, exist_ok=True)
        
    def save(self, kalman_state: dict, cusum_state: dict, current_time: float, equity: float = 0.0, buffer_state: dict = None, feat_history: list = None):
        if current_time - self.last_save < self.interval_sec:
            return False
            
        state = {
            "kalman": kalman_state,
            "cusum": cusum_state,
            "timestamp": current_time,
            "equity": equity
        }
        
        # Use atomic write pattern
        filepath = os.path.join(self.save_dir, "engine_state.json")
        temp_path = filepath + ".tmp"
        
        with open(temp_path, "w") as f:
            json.dump(state, f)
            
        os.replace(temp_path, filepath)
        
        if buffer_state is not None:
            buf_path = os.path.join(self.save_dir, "engine_buffer.npz")
            import numpy as np
            # Save the raw numpy arrays from the buffer
            np.savez_compressed(
                buf_path + ".tmp",
                tns=buffer_state["tns"],
                bids=buffer_state["bids"],
                asks=buffer_state["asks"],
                dts=buffer_state["dts"],
                signs=buffer_state["signs"]
            )
            os.replace(buf_path + ".tmp", buf_path)
            
        if feat_history is not None:
            feat_path = os.path.join(self.save_dir, "feat_history.npz")
            import numpy as np
            if len(feat_history) > 0:
                stacked_feats = np.stack(feat_history, axis=0)
                np.savez_compressed(feat_path + ".tmp", feats=stacked_feats)
                os.replace(feat_path + ".tmp", feat_path)
            
        self.last_save = current_time
        return True
        
    def load(self) -> dict | None:
        filepath = os.path.join(self.save_dir, "engine_state.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r") as f:
            state = json.load(f)
            
        buf_path = os.path.join(self.save_dir, "engine_buffer.npz")
        if os.path.exists(buf_path):
            import numpy as np
            try:
                npz = np.load(buf_path)
                state["buffer_state"] = {
                    "tns": npz["tns"],
                    "bids": npz["bids"],
                    "asks": npz["asks"],
                    "dts": npz["dts"],
                    "signs": npz["signs"]
                }
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Failed to load buffer state: {e}")
                state["buffer_state"] = None
        else:
            state["buffer_state"] = None
            
        feat_path = os.path.join(self.save_dir, "feat_history.npz")
        if os.path.exists(feat_path):
            import numpy as np
            try:
                npz = np.load(feat_path)
                feats = npz["feats"]
                state["feat_history"] = [feats[i] for i in range(feats.shape[0])]
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Failed to load feature history state: {e}")
                state["feat_history"] = None
        else:
            state["feat_history"] = None
            
        return state
