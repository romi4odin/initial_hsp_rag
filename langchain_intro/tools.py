import random
import time

def get_current_time_wait_time(hospital: str) -> int | str :
    """""Dummy function to generte fake times"""

    if hospital not in ["A", "B", "C", "D"]:
        return f"hospital {hospital} doesn't exist"
    
    #simulate API call delay

    time.sleep(1)

    return random.randint(0, 10000)