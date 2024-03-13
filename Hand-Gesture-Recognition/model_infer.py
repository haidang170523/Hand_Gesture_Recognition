import mediapipe as mp
# xử lý thị giác máy tính, trong th này dùng để nhận dạng cử chỉ bàn tay
import joblib
# load model SVM đã được training từ tệp "svm_model.pkl"
import numpy as np # tính toán khoảng cách và biến đổi mảng dữ liệu
import cv2 # xử lý ảnh và video
from timeit import default_timer as timer # đo thời gian thực hiện các phần của mã
mp_hands = mp.solutions.hands # nhận dạng bàn tay
mp_drawing = mp.solutions.drawing_utils # vẽ các đường dẫn trên ảnh
mp_drawing_styles = mp.solutions.drawing_styles


'''
hands là một đối tượng Hand của Mediapipe, được cấu hình với các tham số
cho việc nhận dạng và theo dõi bàn tay, bao gồm độ phức tạp của mô hình,
độ tin cậy tối thiểu cho việc phát hiện và theo dõi, và số lượng tối đa
của bàn tay cần phát hiện
'''
hands = mp_hands.Hands(
    model_complexity=0,#tốc độ cao hơn
    min_detection_confidence=0.5,
    #độ tin cậy: 50% bàn tay sẽ chỉ
    # được coi là phát hiện nếu độ tin cậy ít nhất là 50%, nếu thấp hơn 
    # bàn tay sẽ không đưuọc xử lý-bị bỏ qua
    min_tracking_confidence=0.5,
    # quá trình theo dõi chỉ được tiếp tục nếu
    #độ tin cậy ít nhất là 50%, nếu không quá trình theo dõi cps thể bị dừng
    #hoặc không cập nhật vị trí bàn tay một cách ổn định
    max_num_hands = 1
) 


'''
hàm này sẽ tính và trả về khoảng cách Euclidean
giữa 2 điểm trên mặt phẳng 2D
'''
def euclidean_distance(landmarkA, landmarkB):
    A = np.array([landmarkA.x, landmarkA.y])
    B = np.array([landmarkB.x, landmarkB.y])
    distance = np.linalg.norm(A-B)
    return distance


'''
-hàm này nhận một hình ảnh(hoặc khung video), chuyển đổi n thành định 
dạng RGB và sử dụng MediaPipe để xác định các điểm đánh dấu trên bàn tay
-sau đó, gọi hàm 'add_distance' để tính toán khoảng cách giữa các điểm
đánh dấu và vẽ các đường dẫn trên hình ảnh(nếu có)
-kết quả của một hàm này là một danh sách chứa khoảng cách giữa các điểm
đánh dấu
'''
def process(image, live=True):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    data = []
    #tạo mảng ảnh có cùng kích thước với giá trị bằng 0

    footage = np.zeros(image.shape)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark 
        data = add_distance(landmarks)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                footage,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    else:
        data = [0 for i in range(0, 16)]
    if live:
        cv2.imshow('MediaPipe Hands', cv2.flip(footage, 1))
    return data, footage

'''
chứa các kí tự tương ứng với các cử chỉ bàn tay muốn nhận dạng
'''
characters = ['A',
 'B',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'K',
 'L',
 'M',
 'N',
 'O',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'U',
 'V',
 'W',
 'X',
 'Y',
 'delete',
 'space'
 ]


'''
hàm này tính toán khoảng cách giữa các điểm đánh dấu trên bàn tay
dựa trên danh sách các điểm đánh dấu được truyền vào. Khoảng cách này được tính
bằng hàm 'euclidean_distance'
'''
def add_distance(landmarks):
    data = []
    data.append(euclidean_distance(landmarks[4], landmarks[0]))
    data.append(euclidean_distance(landmarks[8], landmarks[0]))
    data.append(euclidean_distance(landmarks[12], landmarks[0]))
    data.append(euclidean_distance(landmarks[16], landmarks[0]))
    data.append(euclidean_distance(landmarks[20], landmarks[0]))
    data.append(euclidean_distance(landmarks[4], landmarks[8]))
    data.append(euclidean_distance(landmarks[4], landmarks[12]))
    data.append(euclidean_distance(landmarks[8], landmarks[12]))
    data.append(euclidean_distance(landmarks[12], landmarks[16]))
    data.append(euclidean_distance(landmarks[20], landmarks[16]))
    data.append(euclidean_distance(landmarks[8], landmarks[16]))
    data.append(euclidean_distance(landmarks[8], landmarks[20]))
    data.append(euclidean_distance(landmarks[12], landmarks[20]))
    data.append(euclidean_distance(landmarks[4], landmarks[12]))
    data.append(euclidean_distance(landmarks[4], landmarks[16]))
    data.append(euclidean_distance(landmarks[4], landmarks[20]))
    return data

'''
mô hình svm được đào tạo tước tải từ tệp 'svm_model.pkl' bằng
cách sử dụng 'joblib.load'

'''
svm_model = joblib.load('svm_model.pkl')


'''
-input được đẩy vào, sau đó qua vòng lặp while thì các khung hình được 
chụp liên tục từ máy ảnh
-gọi hàm process để xử lý khung và lấy dữ liệu cử chỉ tay
nếu không phát hiện thấy bàn tay nào thì in ra "No hand detect"
. nếu phát hiện thì n sẽ đưa dữ liệu vào mô hình svm để dự đoán
kí tự có xác suất cao nhất sẽ được in, đồng thời hiện thị thời gian thực hiện
và số khung hình trên giây

'''
def hand2sign(image):

    data, footage = process(image, live=False)
    data = np.array(data)
    if np.all(data == 0):
        return "No hand detected", footage
    else:
        classes = svm_model.predict(data.reshape(1, -1))[0]
        return characters[classes], footage
count = 0
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret is None:
            data, footage = process(frame)
            data = np.array(data)
            try:
                if np.all(data == 0):
                    print("No hand detected")
                else:
                    classes = svm_model.predict(data.reshape(1, -1))[0]
                  
                    print(characters[classes])
            except Exception as e:
                print (e)
        cv2.waitKey(1)
