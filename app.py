from flask import Flask, request, render_template, send_file, jsonify, send_from_directory, session, copy_current_request_context
from flask_socketio import SocketIO, join_room, leave_room, close_room, rooms, disconnect
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import concurrent.futures

import io
import os
import argparse
from PIL import Image
import torch
import gc
from peft import PeftModel

import queue
import threading
import uuid
import concurrent.futures
from scripts.process_utils import *

from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
# クライアントIDとルーム情報を保存するグローバル辞書
client_rooms = {}

# レート制限の設定
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# タスクキューの作成とサイズ制限
MAX_QUEUE_SIZE = 100
task_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
active_tasks = {}
task_futures = {}

# ThreadPoolExecutorの作成
executor = concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('MAX_WORKERS', 4)))

gpu_lock = threading.Lock()

class Task:
    def __init__(self, task_id, mode, weight1, weight2, file_data, client_ip, client_id):
        self.task_id = task_id
        self.mode = mode
        self.weight1 = weight1
        self.weight2 = weight2
        self.file_data = file_data
        self.cancel_flag = False
        self.client_ip = client_ip
        self.is_processing = False
        self.client_id = client_id

# キューの状態を通知
def update_queue_status(message):
    socketio.emit('queue_update', {'active_tasks': len(active_tasks), 'message': message}, namespace='/demo')

def process_task(task):
    try:
        task.is_processing = True
        # ファイルデータをPIL Imageに変換
        image = Image.open(io.BytesIO(task.file_data))
        image = ensure_rgb(image)
        
        # キャンセルチェック
        if task.cancel_flag:
            return

        # 画像処理ロジックを呼び出す
        # GPU処理部分
        with gpu_lock:
            sotai_image, sketch_image = process_image_as_base64(image, task.mode, task.weight1, task.weight2)

        # キャンセルチェック
        if task.cancel_flag:
            return
        
        # クライアントIDをリクエストヘッダーから取得（クライアント側で設定する必要があります）
        client_id = task.client_id
        if client_id and client_id in client_rooms:
            room = client_rooms[client_id]
            # ルームにメッセージをemit
            socketio.emit('task_complete', {
                'task_id': task.task_id, 
                'sotai_image': sotai_image, 
                'sketch_image': sketch_image
            }, to=room, namespace='/demo')
        
    except Exception as e:
        print(f"Task error: {str(e)}")
        if not task.cancel_flag:
            client_id = task.client_id
            room = client_rooms[client_id]
            socketio.emit('task_error', {'task_id': task.task_id, 'error': str(e)}, to=room, namespace='/demo')
    finally:
        # タスク数をデクリメント
        client_ip = task.client_ip
        tasks_per_client[client_ip] = tasks_per_client.get(client_ip, 0) - 1
        print(f'Task {task.task_id} completed')
        task.is_processing = False
        if task.task_id in active_tasks.keys():
            del active_tasks[task.task_id]
        if task.task_id in task_futures.keys():
            del task_futures[task.task_id]

        update_queue_status('Task completed or cancelled')

def worker():
    while True:
        try:
            task = task_queue.get()
            if task.task_id in active_tasks.keys():
                future = executor.submit(process_task, task)
                task_futures[task.task_id] = future
        except Exception as e:
            print(f"Worker error: {str(e)}")
        finally:
            # Ensure the task is always removed from the queue
            task_queue.task_done()

# ワーカースレッドの開始
threading.Thread(target=worker, daemon=True).start()

# グローバル変数を使用して接続数とタスク数を管理
connected_clients = 0
tasks_per_client = {}
@socketio.on('connect', namespace='/demo')
def handle_connect(auth):
    client_id = request.sid
    room = f"room_{client_id}"  # クライアントごとに一意のルーム名を生成
    join_room(room)
    client_rooms[client_id] = room
    print(f"Client {client_id} connected and joined room {room}")
    
    global connected_clients
    connected_clients += 1

@socketio.on('disconnect' )
def handle_disconnect():
    client_id = request.sid
    if client_id in client_rooms:
        room = client_rooms[client_id]
        leave_room(room)
        del client_rooms[client_id]
        print(f"Client {client_id} disconnected and removed from room {room}")

    global connected_clients
    connected_clients -= 1
    # キャンセル処理：接続が切断された場合、そのクライアントに関連するタスクをキャンセル。ただし、1番目で処理中のタスクはキャンセルしない
    client_ip = get_remote_address()
    for task_id, task in active_tasks.items():
        if task.client_ip == client_ip and not task.is_processing:
            task.cancel_flag = True
            if task_id in task_futures:
                task_futures[task_id].cancel()
                del task_futures[task_id]
            del active_tasks[task_id]
            tasks_per_client[client_ip] = tasks_per_client.get(client_ip, 0) - 1

@app.route('/submit_task', methods=['POST'])
@limiter.limit("10 per minute")  # 1分間に10回までのリクエストに制限
def submit_task():
    if task_queue.full():
        return jsonify({'error': 'Task queue is full. Please try again later.'}), 503

    # クライアントIPアドレスを取得
    client_ip = get_remote_address()
    # 同一IPからの同時タスク数を制限
    if tasks_per_client.get(client_ip, 0) >= 2:
        return jsonify({'error': 'Maximum number of concurrent tasks reached'}), 429

    task_id = str(uuid.uuid4())
    file = request.files['file']
    mode = request.form.get('mode', 'refine')
    weight1 = float(request.form.get('weight1', 0.4))
    weight2 = float(request.form.get('weight2', 0.3))
    
    # ファイルタイプの制限
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file type'}), 415

    # ファイルデータをバイト列として保存
    file_data = file.read()
    
    client_id = request.headers.get('X-Client-ID')
    task = Task(task_id, mode, weight1, weight2, file_data, client_ip, client_id)
    task_queue.put(task)
    active_tasks[task_id] = task
    
    # 同一IPからのタスク数をインクリメント
    tasks_per_client[client_ip] = tasks_per_client.get(client_ip, 0) + 1

    update_queue_status('Task submitted') # すべてに通知
    
    queue_size = task_queue.qsize()
    task_order = get_active_task_order(task_id)
    return jsonify({'task_id': task_id, 'task_order': task_order, 'queue_size': queue_size}) 

@app.route('/cancel_task/<task_id>', methods=['POST'])
def cancel_task(task_id):
    # クライアントIPアドレスを取得
    client_ip = get_remote_address()

    if task_id in active_tasks.keys():
        task = active_tasks[task_id]
        # タスクの所有者を確認（IPアドレスで簡易的に判断）
        if task.client_ip != client_ip:
            return jsonify({'error': 'Unauthorized to cancel this task'}), 403
        task.cancel_flag = True
        if task_id in task_futures.keys():
            task_futures[task_id].cancel()
            del task_futures[task_id]
        del active_tasks[task_id]
        # タスク数をデクリメント
        tasks_per_client[client_ip] = tasks_per_client.get(client_ip, 0) - 1
        update_queue_status('Task cancelled')
        return jsonify({'message': 'Task cancellation requested'})
    else:
        for task in list(task_queue.queue):
            if task.task_id == task_id and task.client_ip == client_ip:
                task.cancel_flag = True
                # タスク数をデクリメント
                tasks_per_client[client_ip] = tasks_per_client.get(client_ip, 0) - 1
                return jsonify({'message': 'Task cancellation requested for queued task'})
        return jsonify({'error': 'Task not found'}), 404

@app.route('/task_status/<task_id>', methods=['GET'])
def task_status(task_id):
    try:
        if task_id in active_tasks.keys():
            task = active_tasks[task_id]
            return jsonify({'task_id': task_id, 'is_processing': task.is_processing})
        else:
            return jsonify({'task_id': task_id, 'is_processing': False})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_active_task_order(task_id):
    try:
        if task_id not in active_tasks.keys():
            return 0
        if active_tasks[task_id].is_processing:
            return 0
        processing_task_ids = [tid for tid, task in active_tasks.items() if task.is_processing]
        non_processing_task_ids = [tid for tid, task in active_tasks.items() if not task.is_processing]
        if len(processing_task_ids) == 0:
            task_order = 0
        else:
            task_order = non_processing_task_ids.index(task_id) + 1
        return task_order
    except Exception as e:
        print(f"Error getting task order: {str(e)}")

# get_task_orderイベントハンドラー
@app.route('/get_task_order/<task_id>', methods=['GET'])
def handle_get_task_order(task_id):
    if task_id  in active_tasks.keys():
        return jsonify({'task_order': get_active_task_order(task_id)})
    else:
        return jsonify({'task_order': 0})

# Flaskルート
# ルートパスのGETリクエストに対するハンドラ
@app.route('/', methods=['GET'])
def root():
    return render_template("index.html")

# process_refined のエンドポイント
@app.route('/process_refined', methods=['POST'])
def process_refined():
    file = request.files['file']
    weight1 = float(request.form.get('weight1', 0.4))
    weight2 = float(request.form.get('weight2', 0.3))
    
    image = ensure_rgb(Image.open(file.stream))
    sotai_image, sketch_image = process_image_as_base64(image, "refine", weight1, weight2)
    
    return jsonify({
        'sotai_image': sotai_image,
        'sketch_image': sketch_image
    })

@app.route('/process_original', methods=['POST'])
def process_original():
    file = request.files['file']
    
    image = ensure_rgb(Image.open(file.stream))
    sotai_image, sketch_image = process_image_as_base64(image, "original")
    
    return jsonify({
        'sotai_image': sotai_image,
        'sketch_image': sketch_image
    })

@app.route('/process_sketch', methods=['POST'])
def process_sketch():
    file = request.files['file']
    
    image = ensure_rgb(Image.open(file.stream))
    sotai_image, sketch_image = process_image_as_base64(image, "sketch")
    
    return jsonify({
        'sotai_image': sotai_image,
        'sketch_image': sketch_image
    })

# グローバルエラーハンドラー
@app.errorhandler(Exception)
def handle_exception(e):
    # ログにエラーを記録
    app.logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server options.')
    parser.add_argument('--use_local', action='store_true', help='Use local model')
    parser.add_argument('--use_gpu', action='store_true', help='Set to True to use GPU but if not available, it will use CPU')
    parser.add_argument('--use_dotenv', action='store_true', help='Use .env file for environment variables')

    args = parser.parse_args()
    
    initialize(args.use_local, args.use_gpu, args.use_dotenv)

    port = int(os.environ.get('PORT', 7860))
    server = pywsgi.WSGIServer(('0.0.0.0', port), app, handler_class=WebSocketHandler)
    server.serve_forever()
