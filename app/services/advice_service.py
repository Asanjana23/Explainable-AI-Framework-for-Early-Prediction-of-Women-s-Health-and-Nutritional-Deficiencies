def get_advice(anemia, pcos):

    advice = []

    if anemia == 1:
        advice.append("Increase iron intake (spinach, dates, jaggery).")
        advice.append("Consult a doctor for blood test.")

    else:
        advice.append("Maintain healthy diet to prevent anemia.")

    if pcos == 1:
        advice.append("Exercise regularly and maintain weight.")
        advice.append("Avoid junk food and sugar.")
        advice.append("Consult gynecologist.")

    else:
        advice.append("Maintain healthy lifestyle.")

    return advice