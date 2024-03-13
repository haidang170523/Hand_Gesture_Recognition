# Example file showing a basic pygame "game loop"
import time
import pygame
import cv2
import ultis
from model_infer import hand2sign
# pygame setup
# bắt đầu là khởi tạo bằng cách là tạo cửa sổ pygame, khởi tạo đồng hồ kiểm soát khung hình, tạo vòng lặp cho nó
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True

# camera setup, input của chúng ta sẽ được truyền vào từ đây
cap = cv2.VideoCapture(0)

# message and message capturing interval
char = ""
message = ""
# hẹn giờ 3s cho mỗi lần chụp để lấy input đầu vào
CHARACTER_CAPTURE = pygame.USEREVENT + 1
pygame.time.set_timer(CHARACTER_CAPTURE, 3000)

# font
font = pygame.font.SysFont('Arial', 20)
title_font = pygame.font.SysFont('Arial', 50)
# space, delete
blank = 1   

# total message frame
# total_cursor = 0 Khong hieu
total_message = []

timer_interval = 3  # in seconds
start_time = time.time()

# functional alert
delete = False
# bắt đầu chạy trong vòng lặp
while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    try:
        screen.fill('white')
        #khi click vào button X thì chương trình sẽ ngừng chạy
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # nếu không thì sẽ bắt đầu detect
            # nếu sự kiện bắt được chữ cái đk detect thì chữ cái được detect 
            #đó sẽ là chữ cái hiện tại và được lưu vào khung hình hiển
            #thị chữ cái
            elif event.type == CHARACTER_CAPTURE:
                letter = char
                try:
                    # nếu không có bàn tay nào được detect thì space sẽ được in ra
                    # sau space mà vẫn k detect được bàn tay nào thì delete sẽ được thực hiện
                    if letter == "No hand detected":
                        if blank == 1:
                            if not message == "":
                                total_message.append(message)
                                message = " "
                            else:
                                blank = 2
                                delete = True
                        elif blank == 2:
                            if not len(total_message) == 0:
                                total_message.pop()
                                blank = 1
                            delete = False
                        print(total_message)
                    else:
                        message += letter
                        blank = 1
                except Exception as e:
                    pass
        # chụp lại khung hình từ máy ảnh bằng sử dụng đối tượng chụp máy ảnh OpenCV ở cap.read()
        # nếu một khung hình được chụp thành công, nó sẽ tiến hành xử lý khung để phát
        #hiện dấu hiệu bàn tay bằng 'hand2sign' chức năng
        
        ret, frame = cap.read()
        if not ret is None:
            char, footage = hand2sign(frame)
            # kí tự được phát hiện được lưu trong footage, n được lật ngược để đảm bảo định hướng chính xác
            footage = cv2.flip(footage, 1)
            # chuyển đổi mảng Numpy('footage') thành pygame surface
            camera = ultis.np2surface(footage)
            # định vị nguồn cấp dữ liệu camera ở phía bên phải của cửa sổ
            screen.blit(camera, (300, (720-footage.shape[0])/2))
            # print(char)
        
        # hiển thị văn bản tiêu đề có màu đỏ và được đặt phía trên nguồn cấp dữ liệu máy ảnh
        title = title_font.render("Hand sign detection application", True, (255, 0, 0))
        screen.blit(title, (10, (720-footage.shape[0])/2 - title.get_height()*1.5))
        
        # khung tiến trình màu xanh lá cây
        current_time = time.time()
        # thời gian trôi qua
        elapsed_time = current_time - start_time
        # tính toán giá trị tiến trình dưới dạng phần trăm dựa vào thời gian đã trôi qua
        progress = int((elapsed_time % timer_interval) / timer_interval * 100)
        # độ rộng của thanh tiến trình
        progress_width = int(progress / 100 * 280)
        # vẽ thanh tiến trình màu xanh lá cây, bề mặt được vẽ, màu của hình chữ nhật, vị trí là góc trên bên trái hình chữ nhật
        #(x, y, width height)
        pygame.draw.rect(screen, (124, 252, 0), (10, (720-footage.shape[0])/2, progress_width, 50)) # Progress bar
        #hình chữ nhật bao quanh thanh tiến trình tạo đường viền trực quan
        pygame.draw.rect(screen, (124, 252, 0), pygame.Rect(10, (720-footage.shape[0])/2, 300 - 10 - 10, 50), 5) # Progress bar frame
        #khung hình chữ nhật màu đỏ bao quanh khu vực hiển thị tổng tin nhắn
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(300 + footage.shape[1] + 10, 0, 1280 - 300 - footage.shape[1] - 10, 720), 3) # Total message frame
        #khung hình chữ nhật màu đỏ bao quanh khu vực hiển thị tin nhắn tạm thời
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(10, (720-footage.shape[0])/2 + 60, 300 - 10 - 10, 50), 3) # Temporary message frame

        # Drawing message in temporary frame
        temp_msg = font.render(message, True, (255, 0, 0))
        screen.blit(temp_msg, (15, (720-footage.shape[0])/2 + 60 + temp_msg.get_height()/2))
        # Drawing message in total frame
        try:
            for index, msg in enumerate(total_message):
                total_msg = font.render(msg, True, (255, 0, 0))
                line = 10 + index * (10 + total_msg.get_height())
                screen.blit(total_msg, (300 + footage.shape[1] + 15, line))
        except Exception as e:
            print(e)
        # Display instant character
        character = font.render(f"Current char: {char}", True, (255, 0, 0))
        screen.blit(character, (10, (720-footage.shape[0])/2 + 60 + 50 + 10 + 3))

        # Display function state
        space_state = font.render(f"{'Space' if not delete else 'Delete'} is ready!", True, (255, 0, 0))
        screen.blit(space_state, (10, (720-footage.shape[0])/2 + 60 + 50 + 10 + 3 + temp_msg.get_height() + 10))

        pygame.display.flip()
        clock.tick(60)  # Limits FPS to 60
    except Exception as e:
        print(e)
pygame.quit()