import pika
import threading
import logging
import time
import random
import os

# 로그 설정
logging.basicConfig(level=logging.INFO)

# 작업 처리 횟수 및 총 처리 시간 측정을 위한 변수
task_count = 0
total_processing_time = 0
worker_status_interval = 10  # 워커 상태를 기록할 시간 간격 (초)


def dummy_train():
    """
    모델 학습을 시뮬레이션하는 함수입니다. 실제 모델 학습 대신 
    로그 메시지를 출력하고 5초간 대기합니다.
    """

    logging.info("model train start")
    
    # if random.random() < 0.05:
    #     os._exit(1)

    time.sleep(3)
    logging.info("model train finish")


def create_channel():
    """
    RabbitMQ와 연결을 설정하고, 메시지를 주고받을 채널을 생성하는 함수입니다.
    Dead Letter Exchange와 TTL을 설정하여 작업 유실을 방지합니다.

    Returns:
        connection: RabbitMQ 서버와의 연결 객체
        channel: 메시지를 주고받는 채널 객체
    """

    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    # Dead Letter Exchange와 TTL을 설정한 큐 선언
    args = {
        'x-dead-letter-exchange': 'dlx_exchange',  # 메시지가 실패하면 dlx_exchange로 전달
        'x-message-ttl': 180000  # 메시지의 TTL을 3분(180000ms)로 설정
    }

    # durable 옵션을 True 로 사용하여 서버가 재시작되어도 큐와 메시지가 유지되도록 설정
    channel.queue_declare(queue='aiv_train_queue', durable=True, arguments=args)

    # Dead Letter 큐 선언
    channel.queue_declare(queue='dlx_queue', durable=True)
    channel.exchange_declare(exchange='dlx_exchange', exchange_type='direct')
    channel.queue_bind(exchange='dlx_exchange', queue='dlx_queue')

    return connection, channel


def process_task(ch, method, properties, body):
    """
    큐에서 전달된 작업을 처리하는 함수입니다. 작업을 성공적으로 처리하면
    RabbitMQ에 ACK를 보내고, 실패하면 NACK을 보내서 작업을 다시 큐에 넣습니다.

    Args:
        ch: RabbitMQ 채널 객체
        method: 메시지 전달 관련 메타데이터 (delivery_tag 포함)
        properties: 메시지 속성 정보
        body: 큐에서 전달된 메시지 내용 (작업 데이터)
    """
    global task_count, total_processing_time

    start_time = time.time()  # 작업 처리 시작 시간 기록

    try:
        logging.info(f"Processing task: {body.decode()}")
        
        # 더미 학습
        dummy_train()

        # if random.random() < 0.2:
        #     raise Exception("메시지 유실")

        # 작업 성공 시 ACK 전송
        ch.basic_ack(delivery_tag=method.delivery_tag)
        logging.info("Task processed successfully.")

        # 작업이 성공적으로 완료된 경우 처리량 계산
        task_count += 1
        processing_time = time.time() - start_time
        total_processing_time += processing_time
        logging.info(f"Task {task_count} completed in {processing_time:.2f} seconds.")

    except Exception as e:
        logging.error(f"Error processing task: {e}")
        # 실패 시 NACK을 보내고 requeue=True 옵션을 통해 다시 큐에 넣음
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)


def worker_status():
    """
    워커의 상태(가용성, 처리량)를 주기적으로 로그로 기록하는 함수입니다.
    """

    global task_count, total_processing_time
    while True:
        avg_processing_time = total_processing_time / task_count if task_count > 0 else 0
        logging.info(f"Worker status: Tasks processed: {task_count}, Average processing time: {avg_processing_time:.2f} seconds.")
        time.sleep(worker_status_interval)  # 지정한 시간 간격으로 상태 기록


def worker_thread(queue_name):
    """
    RabbitMQ에서 작업을 가져와 처리하는 워커 스레드를 실행하는 함수입니다.
    큐에서 작업을 하나씩 가져와 처리하고, 처리 완료 후 ACK를 전송합니다.
    예외 발생 시 이를 잡아 스레드가 비정상적으로 종료되지 않도록 보호합니다.
    """
    
    try:
        connection, channel = create_channel()

        channel.basic_consume(queue=queue_name, on_message_callback=process_task)

        logging.info("Worker started. Waiting for tasks...")
        channel.start_consuming()

    # 예외 발생 시 스레드가 종료되지 않도록 보호
    except Exception as e:
        logging.error(f"Unexpected error in worker thread: {e}")


def start_workers(worker_count=4):
    """
    여러 개의 워커 스레드를 실행하여 병렬로 작업을 처리할 수 있도록 하는 함수입니다.

    Args:
        worker_count: 실행할 워커 스레드의 개수 (기본값 4)
    """

    # 워커 상태 모니터링 스레드 실행
    status_thread = threading.Thread(target=worker_status)
    status_thread.daemon = True 
    status_thread.start()

    # dlx_queue 처리 스레드
    dlx_thread = threading.Thread(target=worker_thread, args=("dlx_queue",))
    dlx_thread.daemon = True  
    dlx_thread.start()

    threads = []
    for i in range(worker_count):
        thread = threading.Thread(target=worker_thread, args=("aiv_train_queue",))
        thread.daemon = True  # 메인 스레드 종료 시 함께 종료되도록 설정
        threads.append(thread)
        thread.start()

    # 모든 스레드가 종료될 때까지 대기
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    start_workers(worker_count=4)
