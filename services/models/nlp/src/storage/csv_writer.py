import pandas as pd
from pathlib import Path
from typing import Optional
from src.config import get_settings


def save_sentiment_features(df: pd.DataFrame, output_path: Optional[str] = None) -> Optional[str]:
    """
    Saves the aggregated 15-minute sentiment DataFrame sequentially to CSV.
    Safely ignores empty sets and automatically verifies and implements required missing storage directories.
    """
    if df is None or df.empty:
        return None
        
    if output_path is None:
        settings = get_settings()
        # Connect seamlessly back specifically to explicitly requested Config definitions 
        dir_path = settings.OUTPUT_DIR if settings.OUTPUT_DIR else "data/processed"
        output_path = f"{dir_path}/sentiment_features_15m.csv"

    # Explicitly ensure mapping bounds safely exist 
    path_obj = Path(output_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Overwrite output completely to avoid duplicating 15-minute intervals between runs
        df.to_csv(output_path, index=False)
        return str(output_path)
    except Exception as e:
        print(f"Failed extracting mapping into CSV outputs dynamically: {e}")
        return None
