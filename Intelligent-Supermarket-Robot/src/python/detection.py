import csv
import cv2
import os
import sys
import getopt
import signal
import time
from edge_impulse_linux.image import ImageImpulseRunner

runner = None
# if you don't want to see a camera preview, set this to False
show_camera = True
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False

def now():
    return round(time.time() * 1000)

def get_webcams():
    port_ids = []
    for port in range(5):
        print("Looking for a camera in port %s:" % port)
        camera = cv2.VideoCapture(port)
        if camera.isOpened():
            ret = camera.read()[0]
            if ret:
                backendName = camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) found in port %s " % (backendName, h, w, port))
                port_ids.append(port)
            camera.release()
    return port_ids

def sigint_handler(sig, frame):
    print('Interrupted')
    if (runner):
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def help():
    print('python classify.py <path_to_model.eim> <Camera port ID, only required when more than 1 camera is present>')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) == 0:
        help()
        sys.exit(2)

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']
            if len(args) >= 2:
                videoCaptureDeviceId = int(args[1])
            else:
                port_ids = get_webcams()
                if len(port_ids) == 0:
                    raise Exception('Cannot find any webcams')
                if len(args) <= 1 and len(port_ids) > 1:
                    raise Exception("Multiple cameras found. Add the camera port ID as a second argument to use to this script")
                videoCaptureDeviceId = int(port_ids[0])

            camera = cv2.VideoCapture(videoCaptureDeviceId)
            ret = camera.read()[0]
            if ret:
                backendName = camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) in port %s selected." % (backendName, h, w, videoCaptureDeviceId))
                camera.release()
            else:
                raise Exception("Couldn't initialize selected camera.")

            next_frame = now() + 100  # 1 frame per 2 seconds
            last_clear_time = now()  # Initialize the last clear time

            with open('output.csv', 'w', newline='') as csv_file:  # Open output CSV file for writing
                writer = csv.writer(csv_file)
                writer.writerow(['Product ID', 'Label'])

                last_detected_label = None  # Variable to store the last detected label

                for res, img in runner.classifier(videoCaptureDeviceId):
                    if (next_frame > now()):
                        time.sleep((next_frame - now()) / 1000)

                    if "bounding_boxes" in res["result"].keys():
                        for bb in res["result"]["bounding_boxes"]:
                            if bb['value'] > 0.95:
                                label = bb['label']
                                score = bb['value']  # Extract score
                                # Check if the current label is different from the previous one
                                if label != last_detected_label:
                                    if label == "Coke":
                                        writer.writerow(['1', 'Coke'])
                                    elif label == "Lays":
                                        writer.writerow(['2', 'Lays'])
                                    elif label == "Kurkure":
                                        writer.writerow(['3', 'Kurkure'])
                                    print('Detected object: %s (Score: %.2f)' % (label, score))  # Print detected object label and score
                                    img = cv2.rectangle(img, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)
                                    # Update the last detected label
                                    last_detected_label = label
                                    # Update the last clear time
                                    last_clear_time = now()

                    # Clear last detected label if two seconds have passed
                    if now() - last_clear_time >= 2000:
                        last_detected_label = None

                    if (show_camera):
                        cv2.imshow('edgeimpulse', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) == ord('q'):
                            break

                    next_frame = now() + 100  # 1 frame per 5 seconds

        finally:
            if (runner):
                runner.stop()

if __name__ == "__main__":
    main(sys.argv[1:])
