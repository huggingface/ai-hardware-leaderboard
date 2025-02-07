from pydantic import BaseModel
from typing import Optional
from datasets import load_dataset, Dataset
import pandas as pd

class LeaderboardData(BaseModel):
    model_id: str
    backend_type: str
    working: bool
    machine: Optional[str]

def upload_data_to_hub(results: list[LeaderboardData]):
    """
    Upload the results to the hub.
    """
    # Convert results to list of dicts
    results_dict = [result.model_dump() for result in results]
    
    # Download existing dataset
    leaderboard_dataset = load_dataset("hf-hardware/hardware-leaderboard")
    
    # Convert to pandas DataFrame for easier manipulation
    df: pd.DataFrame = leaderboard_dataset.to_pandas()  # type: ignore
    
    # Create DataFrame from new results
    new_df: pd.DataFrame = pd.DataFrame(results_dict)
    
    # Keep track of updated and added entries for commit message
    updated_entries: list[str] = []
    added_entries: list[str] = []
    
    # Update existing entries and add new ones
    # We use model_id and backend_type as unique identifiers
    for _, row in new_df.iterrows():
        row_dict = row.to_dict()
        if row_dict["machine"] == "unknown":
            # do not upload unknown machines to the leaderboard
            continue
        
        mask = (df["model_id"] == row_dict["model_id"]) & (df["backend_type"] == row_dict["backend_type"]) & (df["machine"] == row_dict["machine"])
        entry_desc = f"{row_dict['model_id']} ({row_dict['backend_type']}, {row_dict['machine']})"
        if mask.any():
            # Update existing entry
            row_series = pd.Series(row_dict)
            df.loc[mask] = row_series
            updated_entries.append(entry_desc)
        else:
            # Add new entry
            new_row_df = pd.DataFrame([row_dict])
            df = pd.concat([df, new_row_df], ignore_index=True)
            added_entries.append(entry_desc)
    
    # Create commit message
    commit_parts = []
    if updated_entries:
        commit_parts.append(f"Updated entries: {', '.join(updated_entries)}")
    if added_entries:
        commit_parts.append(f"Added entries: {', '.join(added_entries)}")
    commit_message = " | ".join(commit_parts)
    
    # Convert back to Dataset
    dataset = Dataset.from_pandas(df)
    
    # Push to hub with commit message
    dataset.push_to_hub("hf-hardware/hardware-leaderboard", commit_message=commit_message)
   