def get_risk(prob):

    score = prob[1]

    if score < 0.4:
        return "Low"
    elif score < 0.7:
        return "Medium"
    else:
        return "High"