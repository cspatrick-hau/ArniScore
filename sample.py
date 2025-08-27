from ultralytics import YOLO

def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

model = YOLO("model/object_detection.pt")

results = model.predict(
    source="VID20250719154154.mp4", 
    conf=0.60,
    classes=[0, 1, 2, 3], 
    stream=False, 
    save=True,
    project="runs", 
    name="predictions", 
)

first_contact_winner = None
min_frame = float('inf')  

for frame_idx, result in enumerate(results):
    blue_players = []
    blue_sticks = []
    red_players = []
    red_sticks = []
    
    for box in result.boxes:
        class_id = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        if class_id == 0:    
            blue_players.append((x1, y1, x2, y2))
        elif class_id == 1:
            blue_sticks.append((x1, y1, x2, y2))
        elif class_id == 2: 
            red_players.append((x1, y1, x2, y2))
        elif class_id == 3:
            red_sticks.append((x1, y1, x2, y2))
    
    for red_stick in red_sticks:
        for blue_player in blue_players:
            iou = calculate_iou(red_stick, blue_player)
            if iou > 0:
                print(f"Red Stick hit Blue Player! (IoU={iou:.2f})")
                if first_contact_winner is None:
                    first_contact_winner = "Red Player"
                    print("Red Player scores first!")
    
    for blue_stick in blue_sticks:
        for red_player in red_players:
            iou = calculate_iou(blue_stick, red_player)
            if iou > 0:
                print(f"Blue Stick hit Red Player! (IoU={iou:.2f})")
                if first_contact_winner is None:
                    first_contact_winner = "Blue Player"
                    print("Blue Player scores first!")

if first_contact_winner:
    print(f"\n Final Result: {first_contact_winner} wins!")
else:
    print("No valid hits detected.")