import sys
import cv2
import numpy as np
import pandas as pd
import pyautogui
from PIL import Image
import os
import pickle
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QSpinBox,
    QVBoxLayout, QHBoxLayout, QWidget, QGridLayout,
    QMessageBox, QFileDialog, QCheckBox, QScrollArea
)
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal, QSize, QObject
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap, QImage, QFont

"""
请先运行MhtmlDataExtra程序导出网页卡牌数据,用于该程序的图片识别匹配
"""
class CardRecognizer:
    def __init__(self, card_db_path, error_callback=None):
        self.orb = cv2.ORB_create()
        self.error_callback = error_callback  # 错误回调函数
        self.card_features, self.card_db = self.load_card_database(card_db_path)

    def report_error(self, message):
        """报告错误到回调函数"""
        if self.error_callback:
            self.error_callback(message)

    def compute_image_features(self, image_path):
        if not os.path.exists(image_path):
            self.report_error(f"路径下图像不存在: {image_path}")
            return None
        try:
            pil_img = Image.open(image_path)
            img_array = np.array(pil_img)
            if len(img_array.shape) == 2:
                img_gray = img_array
            else:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            _, des = self.orb.detectAndCompute(img_gray, None)
            return des
        except Exception as e:
            self.report_error(f"路径下图像读取失败: {image_path} - {str(e)}")
            return None

    def load_card_database(self, db_path):
        cache_path = os.path.splitext(db_path)[0] + "_features.pkl"
        try:
            df = pd.read_excel(db_path)
        except Exception as e:
            self.report_error(f"数据加载失败: {str(e)}")
            return {}, pd.DataFrame()  # 返回空数据避免后续错误
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    features_cache = pickle.load(f)
                return features_cache, df
            except Exception as e:
                self.report_error(f"特征缓存加载失败: {str(e)}")
                return {}, pd.DataFrame()
        new_features = {}
        for idx, row in df.iterrows():
            card_name = row['card_name']
            card_path = row['card_path']
            des = self.compute_image_features(card_path)
            if des is not None:
                new_features[card_name] = des
        if new_features:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(new_features, f)
            except Exception as e:
                self.report_error(f"特征保存失败: {str(e)}")
                return {}, pd.DataFrame()
        return new_features, df

    def find_card_match(self, card_image):
        try:
            # 提取截图图像内容
            card_array = np.array(card_image)
            if card_array is None or card_array.size == 0:
                self.report_error("截图图像为空")
                return None
            card_cv = cv2.cvtColor(card_array, cv2.COLOR_RGB2GRAY)
            kp1, des1 = self.orb.detectAndCompute(card_cv, None)
            # 截图图像特征点
            if des1 is None or len(des1) < 3:
                self.report_error("截取图像特征点过少")
                return None
            best_match = None
            best_score = float('inf')
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            for card_name, des2 in self.card_features.items():
                if des2 is None or len(des2) < 3:
                    continue
                try:
                    matches = bf.match(des1, des2)
                    if matches:
                        match_score = sum(m.distance for m in matches) / len(matches)
                        if match_score < best_score:
                            best_score = match_score
                            # 确保返回的是Series中的第一行数据
                            matched_row = self.card_db[self.card_db['card_name'] == card_name]
                            if not matched_row.empty:
                                best_match = matched_row.iloc[0].to_dict()
                except cv2.error as e:
                    self.report_error(f"匹配过程出错: {str(e)}")
                    continue
            return best_match
        except Exception as e:
            self.report_error(f"识别过程崩溃: {str(e)}")
            return None


class SnippingTool(QWidget):
    finished = pyqtSignal(Image.Image, QRect)
    status_message = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)

        screen = QApplication.primaryScreen()
        self.setGeometry(screen.geometry())
        self.setWindowState(Qt.WindowFullScreen)
        self.full_screen = screen.grabWindow(0)
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.is_snipping = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.is_snipping = True
            self.update()
        if event.button() == Qt.RightButton:
            # 发送取消消息
            self.status_message.emit("截图已取消")
            self.close()

    def mouseMoveEvent(self, event):
        if self.is_snipping:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_snipping:
            self.is_snipping = False
            rect = QRect(self.start_point, self.end_point).normalized()
            try:
                if rect.width() < 50 or rect.height() < 50:
                    self.status_message.emit("截图太小，请选择更大区域")
                    self.close()
                    return
                global_rect = QRect(
                    self.mapToGlobal(rect.topLeft()),
                    self.mapToGlobal(rect.bottomRight())
                )
                screenshot = pyautogui.screenshot(
                    region=(
                        global_rect.x(),
                        global_rect.y(),
                        global_rect.width(),
                        global_rect.height()
                    )
                )
                self.finished.emit(screenshot, rect)
            except Exception as e:
                self.status_message.emit(f"截图出错: {str(e)}")
        self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))

        if self.is_snipping:
            rect = QRect(self.start_point, self.end_point).normalized()
            painter.drawPixmap(rect, self.full_screen, rect)
            painter.setPen(QPen(Qt.red, 2))
            painter.drawRect(rect)
            painter.setPen(QPen(Qt.white))
            painter.drawText(
                rect.bottomRight() + QPoint(5, 15),
                f"{rect.width()}x{rect.height()}"
            )
            painter.setPen(QPen(Qt.white))
            painter.drawText(
                QPoint(25, 25),
                "拖动鼠标选择区域 - 右键取消"
            )


class CardThumbnail(QLabel):
    clicked = pyqtSignal(object)
    def __init__(self, pixmap=None, card_info=None, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(128, 128)
        self.setStyleSheet("border: 1px solid gray;")
        self.card_info = card_info
        self.original_pixmap = pixmap
        if pixmap:
            self.update_pixmap()

    def update_pixmap(self):
        """更新缩略图显示"""
        if self.original_pixmap:
            # 限制最大尺寸为256x256
            max_size = QSize(256, 256)
            if self.width() > max_size.width() or self.height() > max_size.height():
                scaled_pixmap = self.original_pixmap.scaled(
                    max_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            else:
                scaled_pixmap = self.original_pixmap.scaled(
                    self.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.update_pixmap()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if self.card_info:
            self.clicked.emit(self.card_info)


class ResultOverlay(QLabel):
    def __init__(self, card_info, region, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("""
            padding: 5px;
            border-radius: 5px;
            font-weight: bold;
        """)
        self.setGeometry(region)
        self.card_info = card_info
        if card_info is not None:
            self.update_display()
        self.setAlignment(Qt.AlignCenter)
        self.adjustSize()

    def update_display(self):
        type_colors = {
            '歌唱': '#FD4C9D',
            '舞蹈': '#3ABAFD',
            '表演': '#FDB10D'
        }
        short_rank = {
            '辅助sp':'SP',
            '特殊':'ES',
            'CT↓':'CT'
        }
        rail_type={
            '无限制':'',
            '红轨':'红',
            '蓝轨':'蓝',
            '黄轨':'黄'
        }
        # 提取主排行
        main_text = str(self.card_info['main_ranks'])
        if main_text != 'nan':
            temp = main_text.split(',') # temp分割后是列表
            if len(temp) == 2:
                text = ''
                for item in temp:
                    text += item.split(':')[1] + '-'
                main_ranks = text[0:-1]
            elif temp[0].split(':')[0] == '对决排行':
                main_ranks = '-' + temp[0].split(':')[1]
            else:
                main_ranks = temp[0].split(':')[1]
        else:
            main_ranks = '-'
        # 提取其他排行
        other_text = str(self.card_info['other_ranks'])
        if other_text:
            temp = other_text.split(',')
            text = ''
            for item in temp:
                t0=str(item.split(':')[1])
                srank = short_rank.get(t0,t0)
                text += srank + '-'
            other_ranks = text[0:-1]
        else:
            other_ranks = "-"

        color = type_colors.get(self.card_info['idol_rarity'])
        idol_symbol = {'得分': '✧', '辅助': '△', '支援': '♡'}.get(self.card_info['idol_type'], '')
        # card_name = self.card_info['card_name'].split('-')[0] + idol_symbol
        # info_text = f"<span style='color:{color}; font-size: 32px;'><b>{card_name}</b></span><br>"
        info_text = f"<span style='color:{color};font-size: 32px;'><b></b> {main_ranks}</span><br>"
        info_text += f"<span style='color:{color};font-size: 32px;'><b></b> {other_ranks}</span><br>"
        """
        悬浮显示没有显示在卡牌中心点,字体大小颜色也需要调整
        """
        # info_text += f"<span style='color:{color};font-size: 32px;'><b></b> {self.card_info['railcolor']}</span>"
        self.setText(info_text)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.close()


class CardStrengthGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IdolPrideRankSystem")
        self.setGeometry(100, 100, 800, 800)
        self.recognizer = None
        self.db_path = ""
        self.card_info = None
        self.selected_region = None
        self.overlays = []
        self.grid_rows = 1
        self.grid_cols = 1
        self.show_overlays = True
        self.show_details = True
        self.init_ui()

    def init_ui(self):
        # 设置整个窗口的默认字体大小
        font = QFont()
        font.setPointSize(12)  # 设置字体大小
        # 应用到整个应用程序
        QApplication.setFont(font)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        # 数据选择区域
        db_layout = QHBoxLayout()
        self.db_label = QLabel("未选择数据")
        db_layout.addWidget(self.db_label)
        # 添加拉伸因子，将后面的控件推到右侧
        db_layout.addStretch(1)
        self.select_db_btn = QPushButton("读取CardRank")
        self.select_db_btn.clicked.connect(self.select_database)
        db_layout.addWidget(self.select_db_btn)
        # 窗口置顶
        self.topmost_check = QCheckBox("置顶")
        self.topmost_check.setChecked(True)
        self.topmost_check.stateChanged.connect(self.toggle_topmost)
        db_layout.addWidget(self.topmost_check)
        main_layout.addLayout(db_layout)
        # 网格设置区域
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("行数:"))
        self.row_spin = QSpinBox()
        self.row_spin.setRange(1, 5)
        self.row_spin.setValue(1)
        grid_layout.addWidget(self.row_spin)
        grid_layout.addWidget(QLabel("列数:"))
        self.col_spin = QSpinBox()
        self.col_spin.setRange(1, 5)
        self.col_spin.setValue(1)
        grid_layout.addWidget(self.col_spin)
        self.overlay_check = QCheckBox("悬浮信息")
        self.overlay_check.setChecked(True)
        self.overlay_check.stateChanged.connect(self.toggle_overlays)
        grid_layout.addWidget(self.overlay_check)
        self.details_check = QCheckBox("详细信息")
        self.details_check.setChecked(True)
        self.details_check.stateChanged.connect(self.toggle_details)
        grid_layout.addWidget(self.details_check)
        main_layout.addLayout(grid_layout)
        # 截图按钮
        self.capture_btn = QPushButton("启动截图识别")
        self.capture_btn.clicked.connect(self.start_snipping)
        self.capture_btn.setEnabled(False)
        main_layout.addWidget(self.capture_btn)
        # 结果展示区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.result_container = QWidget()
        self.result_layout = QGridLayout(self.result_container)
        self.result_layout.setSpacing(10)
        self.scroll_area.setWidget(self.result_container)
        main_layout.addWidget(self.scroll_area)
        # 详细信息标签
        self.details_label = QLabel("请先选择CardRank数据文件")
        self.details_label.setAlignment(Qt.AlignCenter)
        self.details_label.setWordWrap(True)
        main_layout.addWidget(self.details_label)
        self.statusBar().showMessage("就绪")

    def recognizer_error(self, message):
        """处理识别器错误"""
        self.statusBar().showMessage(message)

    def select_database(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择CardRank卡牌排行文件", "", "Excel 文件 (*.xlsx)"
        )
        if file_path:
            self.db_path = file_path
            self.db_label.setText(os.path.basename(file_path))
            try:
                # 创建识别器并传递错误处理回调
                self.recognizer = CardRecognizer(file_path, self.recognizer_error)
                self.capture_btn.setEnabled(True)
                self.statusBar().showMessage(f"卡牌数据加载完成: {len(self.recognizer.card_features)} 张卡牌")
                self.details_label.setText("")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载数据失败: {str(e)}")
                self.statusBar().showMessage("数据加载失败")

    def start_snipping(self):
        if not self.recognizer:
            QMessageBox.warning(self, "警告", "请先选择数据文件")
            return
        # 重置状态
        self.clear_results()
        self.statusBar().showMessage("准备截图...")

        self.grid_rows = self.row_spin.value()
        self.grid_cols = self.col_spin.value()
        self.show_overlays = self.overlay_check.isChecked()
        self.show_details = self.details_check.isChecked()

        self.snipping_tool = SnippingTool(self)
        self.snipping_tool.finished.connect(self.process_screenshot)
        # 连接状态信号
        self.snipping_tool.status_message.connect(self.statusBar().showMessage)
        self.snipping_tool.show()
        self.statusBar().showMessage("按住左键拖动选择区域，右键取消")

    def process_screenshot(self, screenshot, region):
        if screenshot is None:
            self.statusBar().showMessage("区域选择已取消")
            return
        try:
            if screenshot.size[0] < 50 or screenshot.size[1] < 50:
                self.statusBar().showMessage("图像尺寸过小")
                return
            if screenshot.mode != 'RGB':
                screenshot = screenshot.convert('RGB')
            # 清除之前的缩略图和悬浮窗
            self.clear_results()
            # 分割图像为网格
            card_images = self.split_image_grid(screenshot)
            try:
                # 处理每个子图像
                results = []
                for i, card_img in enumerate(card_images):
                    card_info = self.recognizer.find_card_match(card_img)
                    results.append((card_img, card_info))
                    # 显示当前识别进度
                    self.statusBar().showMessage(f"识别中: {i + 1}/{len(card_images)}")
                    QApplication.processEvents()
                # 显示结果
                self.statusBar().showMessage('识别完成')
                self.display_results(results, region)
            except Exception as e:
                self.statusBar().showMessage(f"子图像处理错误: {str(e)}")
        except Exception as e:
            self.statusBar().showMessage(f"图像处理错误: {str(e)}")

    def split_image_grid(self, image):
        width, height = image.size
        card_width = width // self.grid_cols
        card_height = height // self.grid_rows

        card_images = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                left = col * card_width
                upper = row * card_height
                right = left + card_width
                lower = upper + card_height
                card_img = image.crop((left, upper, right, lower))
                card_images.append(card_img)
        return card_images

    def display_results(self, results, region):
        try:
            # 清除之前的布局
            self.clear_results()
            # 使用用户设置的实际行列数
            rows = self.grid_rows
            cols = self.grid_cols
            num_cards = rows * cols  # 确保显示所有单元格
            # 创建缩略图网格
            for idx in range(num_cards):
                row = idx // cols
                col = idx % cols
                # 获取对应的卡片结果（如果存在）
                card_data = results[idx] if idx < len(results) else (None, None)
                card_img, card_info = card_data
                # 优先使用本地图片
                local_pixmap = None

                if card_info and os.path.exists(self.db_path+card_info['card_path']):
                    try:
                        local_pixmap = QPixmap(card_info['card_path'])
                    except:
                        local_pixmap = None
                # 创建缩略图
                if local_pixmap and not local_pixmap.isNull():
                    thumbnail = CardThumbnail(local_pixmap, card_info)
                elif card_img:
                    # 转换为QPixmap
                    if card_img.mode != 'RGB':
                        card_img = card_img.convert('RGB')
                    bytes_per_line = 3 * card_img.size[0]
                    img_data = card_img.tobytes('raw', 'RGB')
                    qimage = QImage(
                        img_data,
                        card_img.size[0],
                        card_img.size[1],
                        bytes_per_line,
                        QImage.Format_RGB888
                    )
                    pixmap = QPixmap.fromImage(qimage)
                    thumbnail = CardThumbnail(pixmap, card_info)
                else:
                    # 创建空缩略图
                    thumbnail = CardThumbnail(None, None)

                # 在添加缩略图到布局之前添加切换信息
                if card_info:
                    thumbnail.clicked.connect(self.show_card_details)
                    
                # 添加到布局
                self.result_layout.addWidget(thumbnail, row, col, Qt.AlignCenter)
                # 创建悬浮信息窗口（如果有卡片信息）
                if card_info and self.show_overlays and isinstance(card_info, dict):
                    # 计算每个卡片的精确位置
                    card_width = region.width() // cols
                    card_height = region.height() // rows
                    overlay_rect = QRect(
                        region.x() + col * card_width,
                        region.y() + row * card_height,
                        card_width,
                        card_height
                    )
                    overlay = ResultOverlay(card_info, overlay_rect, self)
                    overlay.show()
                    self.overlays.append(overlay)
            # 显示第一个卡牌的详细信息
            if results and self.show_details:
                self.show_card_details(results[0][1])
        except Exception as e:
            self.statusBar().showMessage(f"缩略图显示错误: {str(e)}")

    def clear_results(self):
        # 清除之前的悬浮窗
        for overlay in self.overlays:
            overlay.close()
        self.overlays.clear()
        # 清除缩略图
        for i in reversed(range(self.result_layout.count())):
            self.result_layout.itemAt(i).widget().setParent(None)
        self.details_label.setText("")

    def show_card_details(self, card_info):
        if not card_info or not isinstance(card_info, dict):
            self.details_label.setText("未识别到卡牌信息")
            return
        type_colors = {
            '歌唱': '#FD4C9D',
            '舞蹈': '#3ABAFD',
            '表演': '#E77848'
        }
        color = type_colors.get(card_info['idol_rarity'])
        idol_symbol = {'得分': '✧', '辅助': '△', '支援': '♡'}.get(card_info['idol_type'], '')
        card_name = card_info['card_name'].split('-')[0] + '-' + card_info['idol_type'] + idol_symbol
        info_text = f"<span style='color:{color}; font-size: 32px;'><b>{card_name}</b></span><br>"
        info_text += f"<span style='color:{color};font-size: 32px;'><b>主榜:</b> {card_info.get('main_ranks', '无')}</span><br>"
        info_text += f"<span style='color:{color};font-size: 32px;'><b>副榜:</b> {card_info.get('other_ranks', '无')}</span><br>"
        info_text += f"<span style='color:{color};font-size: 32px;'><b>轨道颜色:</b> {card_info.get('railcolor', '无')}</span>"

        self.details_label.setText(info_text)
        self.card_info = card_info

    def toggle_topmost(self, state):
        checked = state == Qt.Checked
        window = self.windowHandle()
        if window:
            if checked:
                window.setFlags(window.flags() | Qt.WindowStaysOnTopHint)
            else:
                window.setFlags(window.flags() & ~Qt.WindowStaysOnTopHint)

    def toggle_overlays(self, state):
        self.show_overlays = state == Qt.Checked
        for overlay in self.overlays:
            overlay.setVisible(self.show_overlays)

    def toggle_details(self, state):
        self.show_details = state == Qt.Checked
        self.details_label.setVisible(self.show_details)

    def closeEvent(self, event):
        # 关闭所有悬浮窗
        for overlay in self.overlays:
            overlay.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CardStrengthGUI()
    window.show()
    sys.exit(app.exec_())