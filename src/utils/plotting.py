import pandas as pd
from mplsoccer import Pitch

def plot_event(df: pd.DataFrame, event_id: str):
    """
    A function to plot a specific match event using an mplsoccer Pitch object.

    Args:
        - events_df (pd.DataFrame): A pandas dataframe of statsbomb match events with 360 data.
        - event_id (str): An ID string for a specific match event.

    Returns:
        p.draw(): An mplsoccer Pitch plot of the specified match event.
    """
    p = Pitch()
    fig, ax = p.draw()
    pitch_data = df[df["id"] == event_id]
    event_type = pitch_data["type"].item().lower()

    p.scatter(
        pitch_data["location"].item()[0],
        pitch_data["location"].item()[1],
        color="purple",
        ax=ax
    )

    p.scatter(
        pitch_data[f"{event_type}_end_location"].item()[0],
        pitch_data[f"{event_type}_end_location"].item()[1],
        color="purple",
        marker="x",
        ax=ax
    )

    p.arrows(
        xstart=pitch_data["location"].item()[0], 
        ystart=pitch_data["location"].item()[1], 
        xend=pitch_data[f"{event_type}_end_location"].item()[0],
        yend=pitch_data[f"{event_type}_end_location"].item()[1],
        ax=ax,
        width=2
    )

    for x in pitch_data["freeze_frame"].item():

        if not x["actor"]:
            if x["teammate"]:
                colour = "skyblue"
            else:
                colour = "salmon"

            p.scatter(x["location"][0], x["location"][1], ax=ax, color=colour)