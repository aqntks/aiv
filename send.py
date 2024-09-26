
import time
import pika

args = {
    'x-dead-letter-exchange': 'dlx_exchange',  # 메시지가 실패하면 dlx_exchange로 전달
    'x-message-ttl': 180000  # 메시지의 TTL을 3분(180000ms)로 설정
}

def send_task():
    """
    큐에 정상적으로 메시지를 전송하는 함수입니다.
    """

    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    # 큐 선언
    channel.queue_declare(queue='aiv_train_queue', durable=True, arguments=args)

    # 메시지 전송
    channel.basic_publish(exchange='',
                          routing_key='aiv_train_queue',
                          body='Model Training')
    print("Sent Model Training")
    connection.close()


def send_bulk_messages(message_count=100):
    """
    다수의 메시지를 큐에 전송하여 부하 테스트를 수행하는 함수입니다.
    """

    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    # 큐 선언
    channel.queue_declare(queue='aiv_train_queue', durable=True, arguments=args)

    for i in range(message_count):
        message = f"Model Training {i}"
        channel.basic_publish(exchange='',
                              routing_key='aiv_train_queue',
                              body=message)
        print(f"Sent {message}")
        time.sleep(0.1) 
    connection.close()



if __name__ == "__main__":
    send_bulk_messages(100)
