import numpy as np
import math
from scipy.spatial import ConvexHull, QhullError

def get_actor_location(freeze_frame: list[dict]) -> list[float] | None:

    for player in freeze_frame:
        if player.get("actor") is True:
            location = player.get("location")

            if isinstance(location, list) and len(location) == 2:
                return location

def get_distance_between_actor_and_player(actor_location: list[float, float], player_location: list[float, float]) -> float:

    if isinstance(player_location, list) and len(player_location) == 2:
        distance = math.dist(actor_location, player_location)

    return distance


def actors_euclidean_distance_to_each_player(freeze_frame: list[dict]) -> dict[str, list[float]] | None:

    if not isinstance(freeze_frame, list):
        return None

    actor_location = get_actor_location(freeze_frame)

    if actor_location is None:
        return None

    teammate_distances = []
    opponent_distances = []

    for player in freeze_frame:

        if player.get("actor") is True:
            continue

        player_location = player.get("location")

        if not isinstance(player_location, list) or len(player_location) != 2:
            continue

        distance = get_distance_between_actor_and_player(
            actor_location,
            player_location
        )

        if player.get("teammate") is True:
            teammate_distances.append(distance)
        else:
            opponent_distances.append(distance)

    return {
        "teammate_distances_to_actor": teammate_distances,
        "opponent_distances_to_actor": opponent_distances
    }



def number_of_players_within_radius_of_actor(freeze_frame: list[dict], radius: float) -> dict[str, int] | None:

    number_of_opponents_within_radius_of_actor = 0
    number_of_teammates_within_radius_of_actor = 0

    actor_location = get_actor_location(freeze_frame)

    if actor_location is None:
        return None

    for player in freeze_frame:

        if player.get("actor") is True:
            continue

        player_location = player.get("location")

        if not isinstance(player_location, list) or len(player_location) != 2:
            continue

        distance = get_distance_between_actor_and_player(
            actor_location,
            player_location
        )

        if distance > radius:
            continue

        if player.get("teammate") is True:
            number_of_teammates_within_radius_of_actor += 1
        else:
            number_of_opponents_within_radius_of_actor += 1

    return {
        "number_of_opponents_within_radius_of_actor": number_of_opponents_within_radius_of_actor,
        "number_of_teammates_within_radius_of_actor": number_of_teammates_within_radius_of_actor
    }



def player_convex_hull_area(freeze_frame: list[dict]) -> dict[str, float] | None:

    opponent_locations = []
    teammate_locations = []

    if not isinstance(freeze_frame, list):
        return None

    for player in freeze_frame:

        player_location = player.get("location")

        if not isinstance(player_location, list) or len(player_location) != 2:
            continue

        if player.get("teammate") is True:
            teammate_locations.append(player_location)
        else:
            opponent_locations.append(player_location)

    try:
        opponent_convex_hull_area = (
            float(ConvexHull(opponent_locations).volume)
            if len(opponent_locations) >= 3
            else np.nan
        )
    except QhullError:
        opponent_convex_hull_area = np.nan

    try:
        teammate_convex_hull_area = (
            float(ConvexHull(teammate_locations).volume)
            if len(teammate_locations) >= 3
            else np.nan
        )
    except QhullError:
        teammate_convex_hull_area = np.nan

    return {
        "opponent_convex_hull_area": opponent_convex_hull_area,
        "teammate_convex_hull_area": teammate_convex_hull_area
    }