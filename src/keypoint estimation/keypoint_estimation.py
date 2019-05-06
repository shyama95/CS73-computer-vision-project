import cv2

proto_file = "mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weights_file = "mpi/pose_iter_160000.caffemodel"
n_points = 15
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13] ]

image = cv2.imread("../../images/input_image1.jpeg")
input_width = image.shape[1]
input_height = image.shape[0]
threshold = 0.1

network = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
input_blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)

network.setInput(input_blob)

output = network.forward()

out_height = output.shape[2]
out_width = output.shape[3]

key_points = []
confidence = []

for i in range(n_points):
    # Confidence map of body part
    probability_map = output[0, i, :, :]

    minimum_value, probability, minimum_location, point = cv2.minMaxLoc(probability_map)
    
    # Scale the point to fit original image
    x = (input_width * point[0]) / out_width
    y = (input_height * point[1]) / out_height

    if probability > threshold :
        confidence.append(probability)
        cv2.circle(image, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Consider keypoints whose probability is greater than set threshold
        key_points.append((int(x), int(y)))
    else:
        key_points.append(None)

# Rearrange keypoints for input to SMPL
key_points = [key_points[10], key_points[9], key_points[8], key_points[11], key_points[12], key_points[13], key_points[4], key_points[3], key_points[2], key_points[5], key_points[6], key_points[7], key_points[1], key_points[0]]
confidence = [confidence[10], confidence[9], confidence[8], confidence[11], confidence[12], confidence[13], confidence[4], confidence[3], confidence[2], confidence[5], confidence[6], confidence[7], confidence[1], confidence[0]]

# Save output image to file
cv2.imwrite('output_keypoints.jpg', image)

# Save keypoints and confidence values to file -> input to SMPL (Generate initial 3D model section)
with open('out_keypoints.txt', 'w') as f:
    f.write(str(key_points))
    f.write('\n')
    f.write(str(confidence))

# Plot keypoints on image
cv2.imshow('Output-Keypoints', image)
cv2.waitKey(0)
