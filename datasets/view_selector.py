def select_camera(question: str) -> str:
    """
    Temporary heuristic camera selector.

    This is only used for baseline pipeline smoke testing.
    It is NOT the final multi-view strategy.
    """

    q = question.lower()

    if "front right" in q:
        return "CAM_FRONT_RIGHT"

    if "front left" in q:
        return "CAM_FRONT_LEFT"

    if "back right" in q or "rear right" in q:
        return "CAM_BACK_RIGHT"

    if "back left" in q or "rear left" in q:
        return "CAM_BACK_LEFT"

    if "behind" in q or "back" in q or "rear" in q:
        return "CAM_BACK"

    return "CAM_FRONT"