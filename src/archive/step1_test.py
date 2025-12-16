import cv2

# --- Input and output paths ---
input_path = "/Users/davidelsweiler/Desktop/test_run.mp4"
output_path = "/Users/davidelsweiler/Desktop/ output_test.mp4"

# --- Open input video ---
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    raise Exception("Could not open video file!")

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Opened: {width}x{height}, {fps} fps")

# --- Choose a processing resolution (faster later for detection) ---
proc_width = 1920
proc_height = int(proc_width * height / width)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_test_step2.mp4", fourcc, fps, (proc_width, proc_height))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Step 2A: downscale frame ---
    frame_small = cv2.resize(frame, (proc_width, proc_height))

    # --- Step 2B: draw a debug overlay ---
    cv2.putText(
        frame_small,
        f"Frame {frame_count}",
        (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        4,
        cv2.LINE_AA
    )

    # Draw a rectangle in the center
    h, w, _ = frame_small.shape
    cv2.rectangle(frame_small, (w//2 - 50, h//2 - 50), (w//2 + 50, h//2 + 50), (255, 0, 0), 4)

    # Write the processed frame
    out.write(frame_small)
    frame_count += 1

cap.release()
out.release()

print(f"Processed {frame_count} frames.")