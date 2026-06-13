# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================
# 必要なライブラリのインポート
# ============================================
import numpy as np
import pycuda.autoinit  # PyCUDAの自動初期化（GPUメモリ管理に必要）
import pycuda.driver as cuda  # CUDA操作用ドライバ
import rclpy  # ROS2のPythonクライアントライブラリ
import tensorrt as trt  # NVIDIA TensorRT（高速推論エンジン）
from cv_bridge import CvBridge  # ROSとOpenCV間の画像変換
from geometry_msgs.msg import Twist, PoseStamped  # 速度コマンドと姿勢メッセージ
from nav_msgs.msg import Odometry, Path  # オドメトリとパス情報
from rclpy.node import Node  # ROS2ノードの基底クラス
from rclpy.time import Time  # ROS2の時刻管理
from sensor_msgs.msg import Image  # カメラ画像メッセージ
from tf2_ros import TransformException  # 座標変換の例外処理
from tf2_ros.buffer import Buffer  # 座標変換バッファ
from tf2_ros.transform_listener import TransformListener  # 座標変換リスナー
from tf2_geometry_msgs import do_transform_pose  # 姿勢の座標変換

# ============================================
# 定数定義（ROSトピック名とパラメータ）
# ============================================
IMAGE_TOPIC_NAME = '/front_stereo_camera/left/image_raw'  # カメラ画像のトピック名
ODOM_TOPIC_NAME = '/chassis/odom'  # オドメトリ（ロボットの速度・位置）のトピック名
CMD_TOPIC_NAME = '/cmd_vel'  # 速度コマンド（制御指令）のトピック名
ROUTE_TOPIC_NAME = '/route'  # ルート情報のトピック名
GOAL_TOPIC_NAME = '/goal_pose'  # ゴール地点のトピック名
PATH_TOPIC_NAME = '/x_mobility_path'  # 計画パスの出力トピック名
RUNTIME_PATH = 'runtime_path'  # TensorRTエンジンファイルのパスパラメータ名
MAPLESS_FLAG = 'is_mapless'  # マップレスモード（地図なし）のフラグ

# ルート処理の定数
NUM_ROUTE_POINTS = 20  # ルートを構成するウェイポイント数
ROUTE_VECTOR_SIZE = 4  # 各ルートベクトルのサイズ（始点x, y + 終点x, y）
ROBOT_FRAME = 'base_link'  # ロボットのベースフレーム（座標系の基準）


# ============================================
# ユーティリティ関数：ルートポイントの補間
# ============================================
def upsample_points(start, goal, max_segment_length):
    """
    始点とゴール間にウェイポイントを補間する関数。
    直線距離が長い場合、指定した最大セグメント長以下になるように中間点を生成。
    
    Args:
        start: 始点の座標 (x, y)
        goal: ゴールの座標 (x, y)
        max_segment_length: セグメント間の最大距離（メートル）
    
    Returns:
        補間されたポイントのリスト [(x1, y1), (x2, y2), ...]
    """
    x1, y1 = start
    x2, y2 = goal

    # 2点間のユークリッド距離を計算
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # 距離が短い場合は補間不要（始点とゴールのみ返す）
    if distance <= max_segment_length:
        return [start, goal]

    # 最大セグメント長に基づいてセグメント数を決定
    # 例: 距離5m、最大1mなら5セグメントに分割
    num_segments = max(1, int(np.ceil(distance / max_segment_length)))

    # 線形補間で中間点を生成（始点から終点まで等間隔）
    interpolated_points = [(x1 + (i / num_segments) * (x2 - x1),
                            y1 + (i / num_segments) * (y2 - y1))
                           for i in range(num_segments + 1)]

    return interpolated_points


# ============================================
# メインクラス：X-Mobilityナビゲーター
# ============================================
class XMobilityNavigator(Node):
    """
    X-Mobility Navigator ROS2ノード
    
    TensorRTを使った深層学習ベースの自律走行システム。
    カメラ画像、オドメトリ、ルート情報を入力として受け取り、
    ニューラルネットワークで車両の制御コマンド（速度・角速度）を推論・出力する。
    
    主な処理フロー:
    1. センサーデータ（画像・速度・ルート）の購読
    2. TensorRTエンジンでの推論実行
    3. 制御コマンドの発行
    """
    def __init__(self):
        super().__init__('x_mobility_navigator')
        
        # ============================================
        # パラメータ宣言（起動時に外部から設定可能）
        # ============================================
        self.declare_parameter(RUNTIME_PATH, '/tmp/x_mobility.engine')  # TensorRTモデルのパス
        self.declare_parameter(MAPLESS_FLAG, True)  # マップレスモード（Trueで地図なし走行）

        # ============================================
        # サブスクライバー（センサーデータの購読）
        # ============================================
        self.image_subscriber = self.create_subscription(
            Image, IMAGE_TOPIC_NAME, self.image_callback, 10)  # カメラ画像
        self.odom_subscriber = self.create_subscription(
            Odometry, ODOM_TOPIC_NAME, self.odom_callback, 10)  # オドメトリ（速度情報）
        self.route_subscriber = self.create_subscription(
            Path, ROUTE_TOPIC_NAME, self.route_callback, 10)  # 計画されたルート
        self.goal_subscriber = self.create_subscription(
            PoseStamped, GOAL_TOPIC_NAME, self.goal_callback, 10)  # ゴール地点

        # ============================================
        # パブリッシャー（制御コマンドとパスの出力）
        # ============================================
        self.cmd_publisher = self.create_publisher(Twist, CMD_TOPIC_NAME, 10)  # 速度コマンド
        self.path_publisher = self.create_publisher(Path, PATH_TOPIC_NAME, 10)  # 可視化用パス

        # ============================================
        # タイマー（定期的な推論実行：0.2秒=5Hz）
        # ============================================
        self.timer = self.create_timer(0.2, self.inference)

        # ============================================
        # TF（座標変換システム）
        # ============================================
        self.tf_buffer = Buffer()  # 座標変換情報を保持するバッファ
        self.tf_listener = TransformListener(self.tf_buffer, self)  # 座標変換をリッスン

        # ============================================
        # 内部状態変数（推論用のデータ保持）
        # ============================================
        # ニューラルネットワークの入出力バッファ
        self.action = np.zeros(6, dtype=np.float32)  # 行動出力（速度・角速度など6次元）
        self.path = np.zeros(10, dtype=np.float32)  # 予測パス（10個の値）
        self.history = np.zeros((1, 1024), dtype=np.float32)  # 時系列の隠れ状態（履歴情報）
        self.sample = np.zeros((1, 512), dtype=np.float32)  # サンプリング用の潜在変数
        
        # センサーデータのバッファ
        self.camera_image = None  # 前処理済みカメラ画像（C×H×W形式）
        self.route_vectors = None  # ルートベクトル（NUM_ROUTE_POINTS-1 × 4）
        self.goal = None  # ゴール地点の姿勢
        self.ego_speed = None  # 自車速度（スカラー値）
        
        # TensorRT推論エンジン関連
        self.runtime_context = None  # TensorRTの実行コンテキスト
        self.stream = cuda.Stream()  # CUDAストリーム（非同期処理用）
        self.cv_bridge = CvBridge()  # 画像変換用

    def load_model(self):
        """
        TensorRTモデルをロードする関数。
        
        処理の流れ:
        1. パラメータからモデルファイルパスを取得
        2. .engineファイル（シリアライズ済みTensorRTモデル）を読み込み
        3. TensorRTランタイムでデシリアライズしてエンジンを作成
        4. 推論用の実行コンテキストを生成
        """
        self.get_logger().info('Loading model')
        # パラメータからモデルファイルパスを取得
        runtime_path = self.get_parameter(
            RUNTIME_PATH).get_parameter_value().string_value
        
        # TensorRTエンジンファイルをバイナリで読み込み
        with open(runtime_path, "rb") as f:
            engine_data = f.read()

        # TensorRTランタイムを作成し、エンジンをデシリアライズ
        runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
        engine = runtime.deserialize_cuda_engine(engine_data)
        # 推論実行用のコンテキストを作成
        self.runtime_context = engine.create_execution_context()

    def image_callback(self, image_msg):
        """
        カメラ画像のコールバック関数。
        受信した画像メッセージを前処理してself.camera_imageに格納。
        """
        self.camera_image = self.process_image_msg(image_msg)

    def odom_callback(self, odom_msg):
        """
        オドメトリのコールバック関数。
        ロボットの現在の前進速度（linear.x）を取得してself.ego_speedに格納。
        """
        self.ego_speed = np.array(odom_msg.twist.twist.linear.x,
                                  dtype=np.float32)

    def goal_callback(self, goal_msg):
        """
        ゴール地点のコールバック関数。
        新しいゴールが設定されたら、履歴状態（history, sample）をリセット。
        これにより、前回のナビゲーションの履歴が新しいタスクに影響しないようにする。
        """
        self.goal = goal_msg
        # 履歴状態をリセット（新しいタスクの開始）
        self.history = np.zeros((1, 1024), dtype=np.float32)
        self.sample = np.zeros((1, 512), dtype=np.float32)

    def route_callback(self, route_msg):
        """
        ルート情報のコールバック関数（マップベースモード用）。
        
        処理の流れ:
        1. ルートの座標系からロボット座標系への変換を取得
        2. ルートポイントをロボット座標系に変換
        3. NUM_ROUTE_POINTS個のポイントを抽出（不足分は最終点で埋める）
        4. 連続する2点をペアにしたルートベクトルを生成
        """
        # ルートの座標系とロボット座標系間の変換を取得
        try:
            transform = self.tf_buffer.lookup_transform(
                ROBOT_FRAME, route_msg.header.frame_id, Time())
        except TransformException as ex:
            self.get_logger().error(
                f'Could not transform {ROBOT_FRAME} to {route_msg.header.frame_id}: {ex}'
            )
            return

        route_poses = route_msg.poses
        num_poses = min(len(route_poses), NUM_ROUTE_POINTS)
        # ルートが空の場合は処理しない
        if num_poses == 0:
            return
        
        # 最初のNUM_ROUTE_POINTSを選択し、不足分は最後のポイントで埋める
        # 例: ルートが15点なら、0-14番目 + 14番目を5回繰り返し → 計20点
        indices = [idx for idx in range(num_poses)]
        indices.extend([num_poses - 1] * (NUM_ROUTE_POINTS - len(indices)))
        
        # 各ルートポイントをロボット座標系に変換してx, yを抽出
        selected_route_positions = []
        for idx in indices:
            transformed_pose = do_transform_pose(route_poses[idx].pose,
                                                 transform)
            selected_route_positions.append(
                [transformed_pose.position.x, transformed_pose.position.y])
        
        # ルートベクトルを生成（各セグメントは [始点x, 始点y, 終点x, 終点y]）
        self.route_vectors = np.zeros(
            (NUM_ROUTE_POINTS - 1, ROUTE_VECTOR_SIZE), np.float32)
        for idx in range(NUM_ROUTE_POINTS - 1):
            self.route_vectors[idx] = np.concatenate(
                (selected_route_positions[idx],
                 selected_route_positions[idx + 1]),
                axis=0)

    def compose_mapless_route(self):
        """
        マップレスモード用のルート生成関数。
        
        地図がない場合、現在位置(0, 0)からゴールまでの直線ルートを生成する。
        
        処理の流れ:
        1. ゴールをロボット座標系に変換
        2. 現在位置とゴール間を1mセグメントに分割（upsample_points）
        3. NUM_ROUTE_POINTS個のポイントを抽出
        4. 連続する2点をペアにしたルートベクトルを生成
        """
        # ゴールが設定されていない場合は処理しない
        if self.goal is None:
            return
        
        # ゴールをロボット座標系に変換
        try:
            transform = self.tf_buffer.lookup_transform(
                ROBOT_FRAME, self.goal.header.frame_id, Time())
        except TransformException as ex:
            self.get_logger().error(
                f'Could not transform {ROBOT_FRAME} to {self.goal.header.frame_id}: {ex}'
            )
            return
        
        goal_in_robot_frame = do_transform_pose(self.goal.pose, transform)
        
        # 現在位置(0, 0)からゴールまでの直線ルートを1mセグメントで補間
        route_poses = upsample_points(
            [0.0, 0.0],  # 現在位置（ロボット座標系の原点）
            [goal_in_robot_frame.position.x, goal_in_robot_frame.position.y],  # ゴール位置
            1.0)  # 最大セグメント長: 1メートル
        
        num_poses = min(len(route_poses), NUM_ROUTE_POINTS)
        # ルートが空の場合は処理しない
        if num_poses == 0:
            return
        
        # 最初のNUM_ROUTE_POINTSを選択し、不足分は最後のポイントで埋める
        indices = [idx for idx in range(num_poses)]
        indices.extend([num_poses - 1] * (NUM_ROUTE_POINTS - len(indices)))
        
        # ルートポイントを抽出
        selected_route_positions = []
        for idx in indices:
            selected_route_positions.append(route_poses[idx])
        
        # ルートベクトルを生成（各セグメントは [始点x, 始点y, 終点x, 終点y]）
        self.route_vectors = np.zeros(
            (NUM_ROUTE_POINTS - 1, ROUTE_VECTOR_SIZE), np.float32)
        for idx in range(NUM_ROUTE_POINTS - 1):
            self.route_vectors[idx] = np.concatenate(
                (selected_route_positions[idx],
                 selected_route_positions[idx + 1]),
                axis=0)

    def inference(self):
        """
        推論メイン関数（タイマーで0.2秒ごとに呼ばれる）。
        
        処理の流れ:
        1. モデルが未ロードなら load_model() を実行
        2. マップレスモードならゴールから直線ルートを生成
        3. 入力データの準備完了チェック
        4. TensorRTで推論実行
        5. 制御コマンドとパスを発行
        """
        # モデルがまだロードされていない場合はロード
        if not self.runtime_context:
            self.load_model()

        # マップレスモードの場合、ゴールから直線ルートを生成
        if self.get_parameter(MAPLESS_FLAG).get_parameter_value().bool_value:
            self.compose_mapless_route()

        # 入力データの準備完了チェック
        # TODO: メッセージの同期処理（タイムスタンプを揃える）
        if self.camera_image is None or self.route_vectors is None or self.ego_speed is None:
            self.get_logger().info(f'Inputs are not ready.')  # データ未準備の警告
            return
        
        # TensorRTで推論実行
        self._trt_inference()
        # 推論結果を発行
        self.publish_action()  # 制御コマンド
        self.publish_path()    # 可視化用パス

    def _trt_inference(self):
        """
        TensorRT推論の実行関数（内部関数）。
        
        処理の流れ:
        1. GPU上に入出力バッファを確保
        2. ホスト（CPU）からデバイス（GPU）へ入力データをコピー
        3. TensorRTエンジンで推論実行
        4. デバイスからホストへ出力データをコピー
        
        入力:
        - カメラ画像
        - ルートベクトル
        - 自車速度
        - 前回の行動（時系列連続性のため）
        - 履歴状態（RNN/LSTMの隠れ状態）
        - サンプリング変数（確率的行動のため）
        
        出力:
        - 行動（速度・角速度コマンド）
        - 予測パス
        - 更新された履歴状態
        - 更新されたサンプリング変数
        """
        # ============================================
        # 1. GPU上にメモリを確保（入力・出力バッファ）
        # ============================================
        image_input = cuda.mem_alloc(self.camera_image.nbytes)  # 画像入力
        route_vec_input = cuda.mem_alloc(self.route_vectors.nbytes)  # ルートベクトル入力
        speed_input = cuda.mem_alloc(self.ego_speed.nbytes)  # 速度入力
        action_input = cuda.mem_alloc(self.action.nbytes)  # 前回の行動入力
        history_input = cuda.mem_alloc(self.history.nbytes)  # 履歴状態入力
        sample_input = cuda.mem_alloc(self.sample.nbytes)  # サンプリング変数入力
        action_output = cuda.mem_alloc(self.action.nbytes)  # 行動出力
        path_output = cuda.mem_alloc(self.path.nbytes)  # パス出力
        history_output = cuda.mem_alloc(self.history.nbytes)  # 履歴状態出力
        sample_ouput = cuda.mem_alloc(self.sample.nbytes)  # サンプリング変数出力

        # ============================================
        # 2. ホスト（CPU）からデバイス（GPU）へデータコピー
        # ============================================
        cuda.memcpy_htod(image_input, self.camera_image)
        cuda.memcpy_htod(route_vec_input, self.route_vectors)
        cuda.memcpy_htod(speed_input, self.ego_speed)
        cuda.memcpy_htod(action_input, self.action)
        cuda.memcpy_htod(history_input, self.history)
        cuda.memcpy_htod(sample_input, self.sample)

        # ============================================
        # 3. バインディングリスト作成（入出力の順序をエンジンに合わせる）
        # ============================================
        # 注意: この順序はTensorRTエンジンの作成時に決まる
        # engine.get_binding_name(binding_idx) で確認可能
        bindings = [
            int(image_input),      # 入力0: 画像
            int(route_vec_input),  # 入力1: ルートベクトル
            int(speed_input),      # 入力2: 速度
            int(action_input),     # 入力3: 前回の行動
            int(history_input),    # 入力4: 履歴状態
            int(sample_input),     # 入力5: サンプリング変数
            int(action_output),    # 出力0: 行動
            int(path_output),      # 出力1: パス
            int(history_output),   # 出力2: 履歴状態
            int(sample_ouput),     # 出力3: サンプリング変数
        ]

        # ============================================
        # 4. TensorRTで推論実行（execute_v2は非同期実行可能）
        # ============================================
        self.runtime_context.execute_v2(bindings)

        # ============================================
        # 5. デバイス（GPU）からホスト（CPU）へ結果をコピー
        # ============================================
        cuda.memcpy_dtoh(self.action, action_output)  # 行動コマンド取得
        cuda.memcpy_dtoh(self.path, path_output)  # パス取得
        cuda.memcpy_dtoh(self.history, history_output)  # 次のステップ用に履歴更新
        cuda.memcpy_dtoh(self.sample, sample_ouput)  # 次のステップ用にサンプル更新

    def publish_action(self):
        """
        推論結果の行動をロボットへ発行する関数。
        
        self.action[0]: 前進速度（m/s）
        self.action[5]: 角速度（rad/s）
        
        他の要素（action[1]～[4]）は加速度や他の制御値の可能性があるが、
        標準的なROS Twistメッセージでは linear.x と angular.z のみ使用。
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = float(self.action[0])  # 前進速度
        cmd_vel.angular.z = float(self.action[5])  # 角速度（左旋回が正）
        self.cmd_publisher.publish(cmd_vel)

    def publish_path(self):
        """
        可視化用にルートベクトルをPathメッセージとして発行する関数。
        
        RVizなどのツールでロボットが従うべきルートを可視化できる。
        各ルートベクトルの始点（x, y）を抽出してパスとして構成。
        """
        path = Path()
        path.header.frame_id = ROBOT_FRAME  # ロボット座標系
        path.header.stamp = self.get_clock().now().to_msg()  # 現在時刻
        
        # 各ルートベクトルの始点をパスポイントとして追加
        for idx in range(len(self.route_vectors)):
            path_pose = PoseStamped()
            path_pose.header = path.header
            # route_vectors[idx][0:2] は始点の (x, y)
            path_pose.pose.position.x = float(self.route_vectors[idx][0])
            path_pose.pose.position.y = float(self.route_vectors[idx][1])
            path.poses.append(path_pose)
        
        self.path_publisher.publish(path)

    def process_image_msg(self, image_msg):
        """
        ROSの画像メッセージを前処理する関数。
        
        処理内容:
        1. 1次元配列をH×W×C形式に再構成
        2. C×H×W形式に転置（PyTorch/TensorRTの標準形式）
        3. [0, 255] → [0.0, 1.0] に正規化
        4. メモリ連続配列に変換（CUDA転送の効率化）
        
        Args:
            image_msg: ROS Image メッセージ
        
        Returns:
            前処理済み画像（C×H×W、float32、[0.0, 1.0]）
        """
        # チャンネル数を計算（stepは1行のバイト数）
        image_channels = int(image_msg.step / image_msg.width)
        
        # 1次元配列を3次元（H×W×C）に再構成
        image = np.array(image_msg.data).reshape(
            (image_msg.height, image_msg.width, image_channels))
        
        # H×W×C → C×H×W に転置し、正規化（0-255 → 0.0-1.0）
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # メモリ連続配列に変換（CUDA転送の高速化）
        return np.ascontiguousarray(image)


# ============================================
# メイン関数（プログラムのエントリポイント）
# ============================================
def main(args=None):
    """
    ROS2ノードの起動関数。
    
    処理の流れ:
    1. ROS2の初期化
    2. XMobilityNavigatorノードのインスタンス作成
    3. spin（メッセージ受信ループ）の開始
    """
    rclpy.init(args=args)  # ROS2の初期化
    x_mobility_navigator = XMobilityNavigator()  # ノードのインスタンス作成
    rclpy.spin(x_mobility_navigator)  # コールバック処理を開始（Ctrl+Cまで継続）


if __name__ == '__main__':
    main()  # スクリプトとして実行された場合のみmain()を呼ぶ