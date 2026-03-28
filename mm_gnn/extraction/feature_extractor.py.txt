import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import easyocr
import io

class MultiModalExtractor:
    def __init__(self):
        print("Initializing MultiModal Extractor (Loading Models)...")
        # Automatically use GPU if available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Load Scene Text Extractor (OCR)
        self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # 2. Load Visual Object Extractor (Faster R-CNN)
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.object_detector = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
        self.object_detector.to(self.device)
        self.object_detector.eval() # Set to evaluation mode

    def _normalize_bboxes(self, boxes, img_width, img_height):
        """
        Converts standard [x_min, y_min, x_max, y_max] boxes into the 10-dimensional 
        features specified in Paper Section 4.1.
        """
        num_boxes = boxes.shape[0]
        bbox_features = torch.zeros((num_boxes, 10))
        
        for i in range(num_boxes):
            xmin, ymin, xmax, ymax = boxes[i]
            
            # Normalize coordinates to [0, 1]
            xmin, xmax = xmin / img_width, xmax / img_width
            ymin, ymax = ymin / img_height, ymax / img_height
            
            w = xmax - xmin
            h = ymax - ymin
            x_center = xmin + (w / 2)
            y_center = ymin + (h / 2)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0.0
            
            # 10-D tensor: [center_x, center_y, x_min, y_min, x_max, y_max, w, h, area, aspect_ratio]
            bbox_features[i] = torch.tensor([
                x_center, y_center, xmin, ymin, xmax, ymax, w, h, area, aspect_ratio
            ])
            
        return bbox_features

    def extract(self, image_bytes, question=""):
        """
        Processes the image and returns multi-modal nodes and bounding boxes.
        """
        # Load image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_width, img_height = img.size
        img_tensor = F.to_tensor(img).unsqueeze(0).to(self.device)

        # ==========================================
        # A. EXTRACT SEMANTIC/TEXT FEATURES (OCR)
        # ==========================================
        ocr_results = self.ocr_reader.readtext(img)
        extracted_texts = []
        raw_s_boxes = []
        
        for result in ocr_results:
            # EasyOCR returns bounding box as 4 points: [top-left, top-right, bottom-right, bottom-left]
            coords = result[0]
            xmin = min([pt[0] for pt in coords])
            ymin = min([pt[1] for pt in coords])
            xmax = max([pt[0] for pt in coords])
            ymax = max([pt[1] for pt in coords])
            
            raw_s_boxes.append([xmin, ymin, xmax, ymax])
            extracted_texts.append(result[1])

        # Failsafe: If no text is found, inject a dummy node so the graph doesn't crash
        if len(extracted_texts) == 0:
            extracted_texts = ['STOP']
            raw_s_boxes = [[0.0, 0.0, 10.0, 10.0]]
            
        s_bboxes = self._normalize_bboxes(torch.tensor(raw_s_boxes), img_width, img_height)

        # ==========================================
        # B. EXTRACT VISUAL FEATURES (Faster R-CNN)
        # ==========================================
        with torch.no_grad():
            detection_results = self.object_detector(img_tensor)[0]

        # Filter high-confidence objects
        high_conf_indices = detection_results['scores'] > 0.8
        visual_boxes = detection_results['boxes'][high_conf_indices]

        # Failsafe: Ensure we have at least one visual object
        if len(visual_boxes) == 0:
            visual_boxes = torch.tensor([[0.0, 0.0, img_width, img_height]])
            
        v_bboxes = self._normalize_bboxes(visual_boxes, img_width, img_height)

        # ==========================================
        # C. INITIALIZE NODE FEATURES
        # Note: In a full production setup, text would pass through FastText 
        # and boxes would crop ResNet feature maps. We simulate the dimensions here.
        # ==========================================
        num_visual = len(visual_boxes)
        num_semantic = len(extracted_texts)
        
        # Dimensions specified by the paper (Visual=2048, Semantic=300, Question=512)
        v_nodes = torch.rand((num_visual, 2048))
        s_nodes = torch.rand((num_semantic, 300))
        q_feat = torch.rand((512,))

        return {
            "visual": v_nodes,
            "semantic": s_nodes,
            "numeric": None, # Kept as None for baseline implementation
            "v_bboxes": v_bboxes,
            "s_bboxes": s_bboxes,
            "question": q_feat,
            "raw_texts": extracted_texts
        }