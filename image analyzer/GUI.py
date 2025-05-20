from PyQt5.QtWidgets import QListWidget,QSplitter, QApplication, QBoxLayout,QHBoxLayout,QWidget,QMainWindow, QAction, QFileDialog, QLabel, QVBoxLayout, QFrame, QMenu, QInputDialog, QToolBar
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt
import cv2,os,piexif, sys,filetype
from PIL import Image,ImageChops, ImageEnhance
from PIL.ExifTags import TAGS
import DB,Lowes,ORB_BF,ORBFLANN,SC,DCT
from matplotlib import pyplot as plt 
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from PyQt5.QtWidgets import QMessageBox,QPushButton
from PyQt5.QtGui import QTransform
from functools import partial


class CustomVSCodeStyleGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.cv_image = None  
        self.pixmap = None 
        self.zoom_scale = 1.0
        self.showing_forged_image = False

        self.setWindowTitle("Image Analyzer")
        self.setGeometry(100, 100, 1200, 700)

        # Central Widget and Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        self.subplot = QToolBar("SubMenuToolbar", self); self.subplot.setVisible(False); self.addToolBar(Qt.TopToolBarArea, self.subplot)
        # Sidebar with icons
        sidebar = QVBoxLayout()
        open_icon = QPushButton()
        open_icon.setIcon(QIcon("Open.jpeg"))
        open_icon.clicked.connect(self.openImage)

        quality_icon = QPushButton()
        quality_icon.setIcon(QIcon("Quality_Analyis.png"))
        quality_icon.clicked.connect(lambda: self.load_subtools(['Blur Analyis', 'Histogram','Grey Histogram']))

        imapoe_icon = QPushButton()
        imapoe_icon.setIcon(QIcon("Image_Processing.png"))
        imapoe_icon.clicked.connect(lambda: self.load_subtools(['ROI Extraction','Error Level Analysis']))

        for btn in [open_icon, quality_icon, imapoe_icon]:
            btn.setFixedSize(40, 40)
            sidebar.addWidget(btn)
        sidebar.addStretch()

        # Splitter Area
        # Splitter Area
        splitter = QSplitter(Qt.Horizontal)

# Subtool area (now on the left)
        self.subtool_list = QListWidget()
        self.subtool_list.setVisible(False)
        self.subtool_list.setMaximumWidth(200)
        self.subtool_list.itemClicked.connect(self.handle_subtool_selection)


# Right panel: image area and result area
        right_panel = QSplitter(Qt.Vertical)

        self.imageLabel = QLabel("Image Area")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setFrameShape(QFrame.StyledPanel)

        self.resultLabel = QLabel("Result Area")
        self.resultLabel.setAlignment(Qt.AlignCenter)
        self.resultLabel.setFrameShape(QFrame.StyledPanel)

        right_panel.addWidget(self.imageLabel)
        right_panel.addWidget(self.resultLabel)

# Add to splitter in new order
        splitter.addWidget(self.subtool_list)
        splitter.addWidget(right_panel)

# Final layout
        main_layout.addLayout(sidebar)
        main_layout.addWidget(splitter)

        menubar = self.menuBar()
        self.menus = {
            "file": menubar.addMenu("File"),
            "edit": menubar.addMenu("Edit"),
            "view": menubar.addMenu("View"),
            "forgery": menubar.addMenu("Forgery"),
            "About": menubar.addMenu("About")
        }
        for name, menu in self.menus.items():
            menu.aboutToShow.connect(partial(self.display_toolbar, name))

        passive_menu = self.menus["forgery"].addMenu("Passive")

        orb=passive_menu.addMenu("ORB")
        orbBFAction = QAction('ORB (BF Matcher)', self)
        orbBFAction.triggered.connect(self.ORB_BF)
        orb.addAction(orbBFAction)

        orbFLANNAction = QAction('ORB (FLANN Matcher)', self)
        orbFLANNAction.triggered.connect(self.ORB_FLANN)
        orb.addAction(orbFLANNAction)
        
        
# Add actions to submenus
        # passive_menu.addAction("Detect Copy-Move", self.saveImageAs)
        subMenuItems = ['SIFT with DBSCAN', 'Lowes Ratio', 'Sensitivity Clustering', 'Discrete Cosine Transform', 'Discrete Fourier Transform ', 'Fast Fourier Transform']
        for item in subMenuItems:
            
            action = QAction(item, self)
            if item == 'SIFT with DBSCAN':
                action.triggered.connect(self.DBSHOW)
            elif item == 'Lowes Ratio':
                action.triggered.connect(self.lowes_ratio_action)
            elif item == 'Sensitivity Clustering':
                action.triggered.connect(self.SensitivityClustering)
            elif item == 'Discrete Cosine Transform':
                action.triggered.connect(self.DCTCheck)
            passive_menu.addAction(action)
        
        self.actions = {
            "file": [("Open", self.openImage), ("Save", self.saveImage), ("Save As", self.saveImageAs), ("Exit", self.close)],
            "edit": [("Refresh", self.refreshImage), ("Rotate image",self.rotateImage),("Resize",self.resizeImage),("EXIF Details", self.exifimage)],
            "view": [("Zoom in", self.zoomIn),("Zoom out", self.zoomOut)],
            "About": [("About This App", self.showAbout)]
    

        }
    def handle_subtool_selection(self, item):
        tool_name = item.text()

        if tool_name == 'Blur Analyis':
            self.blurAnalysisSelected()
        elif tool_name == 'Histogram':
            self.showHistogramSelected()
        elif tool_name == 'Grey Histogram':
            self.showGrayHistogram()
        elif tool_name == 'ROI Extraction':
            self.roiExtractionSelected()
        elif tool_name == 'Error Level Analysis':
            self.errorLevelSelected()
        else:
            QMessageBox.information(self, 'Tool Not Implemented', f'{tool_name} functionality not implemented yet.')

    def showAbout(self):
        QMessageBox.information(self, 'About', 'This software is used for detecting image forgeries.')

    def display_toolbar(self,menu_type):
        self.subplot.clear(); self.subplot.setVisible(True)
        for label, handler in self.actions.get(menu_type, []):
            action = QAction(label, self)
            action.triggered.connect(handler)
            self.subplot.addAction(action)
    def saveImage(self):
        if self.cv_image is not None:
            path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.bmp)")
            if path: cv2.imwrite(path, self.cv_image); QMessageBox.information(self, 'Save Image', 'Image saved successfully.')
        else:
            QMessageBox.warning(self, "Error", "No image opened")
    def displayImage(self):
        if self.cv_image is not None:
            rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(convert_to_Qt_format)
            self.imageLabel.setPixmap(self.pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio))
            self.resultLabel.setText("Image opened: " + os.path.basename(self.image_path))
        else:
            self.resultLabel.setText("Error: No image opened")
    def saveImageAs(self):
        if self.pixmap:
            path, _ = QFileDialog.getSaveFileName(self, "Save Image As", "", "Images (*.png *.jpg *.bmp)")
            if path: self.pixmap.save(path); QMessageBox.information(self, 'Save Image As', 'Image saved successfully.')

    def get_icon(self, name, fallback):
        icon = QIcon.fromTheme(name)
        return icon if not icon.isNull() else QIcon(fallback)

    def roiExtractionSelected(self):
        if self.image_path:
            image = cv2.imread(self.image_path)
            original = image.copy()
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            
            if cnts:
                cv2.drawContours(mask, [cnts[0]], -1, 255, -1)
                close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
                self.result = cv2.bitwise_and(original, original, mask=close)
                self.result[close == 0] = (0, 0, 0)
                self.cv_image = self.result.copy()
                self.displayImage()
            else:
                QMessageBox.warning(self, "ROI Error", "No regions detected.")

    def refreshImage(self):
        self.imageLabel.clear()
        self.resultLabel.clear()
        self.cv_image = None
        self.pixmap = None
        self.zoom_scale = 1.0
        self.image_path = None
    def rotateImage(self):
        if self.cv_image is not None:
            angle, ok = QInputDialog.getDouble(self, 'Rotate Image', 'Angle (Degree):')
            if ok:
                try:
                # Create a transformation matrix for rotation
                    transform = QTransform().rotate(angle)
                
                # Rotate the image using the transformation matrix
                    rotated_pixmap = self.pixmap.transformed(transform)
                    self.imageLabel.setPixmap(rotated_pixmap)
                
                except ValueError:
                     QMessageBox.warning(self, "Error", "Invalid angle format")
        else:
            QMessageBox.warning(self, "Error", "No image opened")
    def exifimage(self):
        if self.image_path:
            filename = self.image_path
            image_info = get_image_info(filename)
            if image_info:
                info_str = "\n".join(f"{key}: {value}" for key, value in image_info.items())
                self.resultLabel.setText(info_str)
            else:
                self.resultLabel.setText('Failed to retrieve image information.')
        else:
            self.resultLabel.setText('No image opened.')

    def resizeImage(self):
        if self.cv_image is not None:
            size, ok = QInputDialog.getText(self, 'Resize Image', 'size (height, weight):')
            if ok:
                try:
                    width, height = map(int, size.split(','))
                    self.cv_image = cv2.resize(self.cv_image, (width, height))
                    self.displayImage()
                except ValueError:
                    QMessageBox.warning(self, "Error", "Invalid size format")
        else:
            QMessageBox.warning(self, "Error", "No image opened")

    def zoomIn(self):
        if self.pixmap:
            self.zoom_scale *= 1.25
            self.applyZoom()
        else:
            QMessageBox.warning(self, "Error", "No image opened")

    def zoomOut(self):
        if self.pixmap:
            self.zoom_scale /= 1.25
            self.applyZoom()
        else:
            QMessageBox.warning(self, "Error", "No image opened")

    def applyZoom(self):
        size = self.pixmap.size() * self.zoom_scale
        scaled_pixmap = self.pixmap.scaled(size, Qt.KeepAspectRatio)
        self.imageLabel.setPixmap(scaled_pixmap)
        self.imageLabel.adjustSize()
    def blurAnalysisSelected(self):
        if self.cv_image is not None:
            gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            threshold = 100  # Adjust this threshold based on experimentation
            if laplacian_var < threshold:
                QMessageBox.information(self, "Blur Analysis", "Image is slightly blurred. Attempting to deblur.")
                deblurred = self.wiener_deblur(gray)
                if deblurred is not None:
                    self.cv_image = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)
                    self.displayImage()
                else:
                    QMessageBox.warning(self, "Deblur Error", "Deblurring failed.")
            else:
                QMessageBox.information(self, "Blur Analysis", "Image is sharp.")
        else:
            QMessageBox.warning(self, "Error", "No image loaded.")

    def wiener_deblur(self, img):
    # Define the point spread function (PSF)
        psf = np.ones((5, 5)) / 25
        img_fft = np.fft.fft2(img)
        psf_fft = np.fft.fft2(psf, s=img.shape)
        psf_fft_conj = np.conj(psf_fft)
        psf_fft_abs2 = np.abs(psf_fft) ** 2
        eps = 1e-3  # Regularization parameter
        wiener_filter = psf_fft_conj / (psf_fft_abs2 + eps)
        result_fft = img_fft * wiener_filter
        result = np.fft.ifft2(result_fft)
        result = np.abs(result)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    def showGrayHistogram(self):
        if self.cv_image is not None:
            gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            plt.figure()
            plt.title("Grayscale Histogram")
            plt.xlabel("Gray Level")
            plt.ylabel("Frequency")
            plt.plot(hist)
            plt.xlim([0, 256])
            plt.show()
        else:
            QMessageBox.warning(self, "Error", "No image opened")
            
    def showHistogramSelected(self):
        if self.image_path:
            image = self.cv_image
            channels = cv2.split(image)
            colors = ('b', 'g', 'r')
            channel_names = ('Blue', 'Green', 'Red')
            plt.figure(figsize=(10, 5))
            for channel, color, name in zip(channels, colors, channel_names):
                histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
                plt.plot(histogram, color=color, label=name)

            plt.title('Color Histogram')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            plt.show()

        else:
            QMessageBox.warning(self, "Error", "No image opened")
    def openImage(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.image_path = file_path  # Fix added
            self.cv_image = cv2.imread(file_path)
            self.pixmap = QPixmap(file_path).scaled(500, 500, Qt.KeepAspectRatio)
            self.imageLabel.setPixmap(self.pixmap)
            self.resultLabel.setText(f"Image opened: {os.path.basename(file_path)}")

            
    def load_subtools(self, tools):
        self.subtool_list.setVisible (True)
        self.subtool_list.clear()
        self.subtool_list.addItems(tools)
    def SensitivityClustering(self):
        if self.image_path:
            if self.showing_forged_image:  # Check if forged image is being displayed
                self.cv_image = cv2.imread(self.image_path)  # Load the original image
                self.showing_forged_image = False  # Set flag to False
                self.updateImageLabel()  # Update the image label to show the original image

            self.resultLabel.clear()
            min_cluster_size, ok = QInputDialog.getInt(self, 'Select Threshold', 'Best range:750-10000 ')
            if ok:
                result = SC.detect_copy_move(self.image_path, int(min_cluster_size))
                self.resultLabel.setText(f"According Sensitivity Clustering Method \n\n Image is assumed to be: {result}\t with the threshold {min_cluster_size}")
        else:
            QMessageBox.warning(self, "Error", "No image opened")
    def lowes_ratio_action(self):
        if self.image_path:
            self.resultLabel.clear()
            threshold, ok = QInputDialog.getDouble(self, 'Select Threshold', 'Best range:1.8 - 4.0 ')

            if ok:
                result = Lowes.lowes_ratio(self.image_path,float(threshold))
                self.resultLabel.setText(f"According Lowe's Ratio Method \n\n Image is assumed to be: {result}\t with the threshold {threshold}")
        else:
            QMessageBox.warning(self, "Error", "No image opened")
    def ORB_BF(self):
        if self.image_path:
            self.resultLabel.clear()
            min_matches, ok = QInputDialog.getInt(self, 'Select Threshold', 'Best range: 5 - 35 ')
            if ok:
                result = ORB_BF.detect_orb_bf(self.image_path,float(min_matches))
                self.resultLabel.setText(f"According ORB with BF matcher Method \n\n Image is assumed to be: {result}\t with the threshold {min_matches}")
        else:
            QMessageBox.warning(self, "Error", "No image opened")
    def ORB_FLANN(self):
        if self.image_path:
            self.resultLabel.clear()
            min_matches, ok = QInputDialog.getInt(self, 'Select Threshold', 'Best range: 5 - 35 ')
            if ok:
                result = ORBFLANN.detect_orb_flann(self.image_path,float(min_matches))
                self.resultLabel.setText(f"According ORB with FLANN Matcher Method \n\n Image is assumed to be: {result}\t with the threshold {min_matches}")
        else:
            QMessageBox.warning(self, "Error", "No image opened")
    def DCTCheck(self):
        if self.image_path:
            self.resultLabel.clear()
            threshold, ok = QInputDialog.getInt(self, 'Select Threshold', 'Best range: 50 - 400 ')
            if ok:
                result = DCT.detect_copy_move(self.image_path, int(threshold))
                self.resultLabel.setText(f"According DCT Method \n\n Image is assumed to be: {result}\t with the threshold {threshold}")
        else:
            QMessageBox.warning(self, "Error", "No image opened")
    
    def DBSHOW(self):
        if self.image_path:
            self.resultLabel.clear()
            image_path = self.image_path
            eps, ok = QInputDialog.getInt(self, 'Select Threshold', 'Best range: 19 - 60 ')
            if ok:
                key_points, descriptors = DB.siftDetector(image_path)
                forgery_status, forgery_image = DB.ShowForgery(image_path, key_points, descriptors,int(eps))
                
                if forgery_status == "Forged":
                    self.cv_image = forgery_image.copy()
                    self.updateImageLabel()
                    self.showing_forged_image = True
                    self.zoom_scale = 1
                    self.pixmap = None
                else:
                    self.showAlertMessage("No Forgery Detected", "The image is original.")
        else:
            self.showAlertMessage("Error", "No image opened.")
    def updateImageLabel(self):
        if self.cv_image is not None:
            height, width, channel = self.cv_image.shape
            bytesPerLine = channel * width
            qImg = QImage(self.cv_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.imageLabel.setPixmap(pixmap)
           # self.imageLabel.setAlignment(Qt.AlignCenter)
          #  self.imageLabel.setScaledContents(True)
    def errorLevelSelected(self):
        if self.image_path:
            original = Image.open(self.image_path).convert('RGB')
            resaved_path = "resaved.jpg"
            original.save(resaved_path, 'JPEG', quality=90)
            resaved = Image.open(resaved_path)
            ela_image = ImageChops.difference(original, resaved)
            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                QMessageBox.information(self, "ELA", "No differences detected.")
                return
            scale = 255.0 / max_diff
            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
            ela_image.save("ela_result.jpg")
        # Load and display the ELA result
            ela_cv = cv2.imread("ela_result.jpg")
            self.cv_image = ela_cv
            self.displayImage()
        # Clean up temporary files
            os.remove(resaved_path)
            os.remove("ela_result.jpg")
        else:
            QMessageBox.warning(self, "Error", "No image loaded.")
def get_image_info(filename):
    try:
        img = Image.open(filename)
        return {
            "file_name": os.path.basename(filename),
            "file_size": os.path.getsize(filename),
            "image_size": img.size,
            "file_type": img.format,
            "mime_type": Image.MIME.get(img.format, 'Unknown')

        }
    except IOError:
        return None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CustomVSCodeStyleGUI()
    window.show()
    sys.exit(app.exec_())
