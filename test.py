from datetime import datetime

present_to_away_time = 2
away_to_present_time = 0.7 

now_time = datetime.now()

results = True

current_state = False
candidate_state = None
candidate_since = None

if(results):
    detected_state = True

else:
    detected_state = False


if (detected_state == current_state):
    candidate_state = None
    candidate_since = None

else:
    if ( detected_state != current_state):
        if