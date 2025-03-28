from typing import Tuple

def get_risk_level(risk: float) -> Tuple[str, str]:
    if risk < 0.1:
        return "низький", "green"
    elif risk < 0.23:
        return "помірний", "yellowgreen"
    elif risk < 0.4:
        return "середній", "orange"
    elif risk < 0.56:
        return "високий", "orangered"
    elif risk < 0.78:
        return "дуже високий", "red"
    else:
        return "дуже високий", "red"
