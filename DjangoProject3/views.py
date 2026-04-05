from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageSerializer
import torch
from torchvision import transforms, models
from PIL import Image
import io
from torch import nn
import numpy as np
import torch.nn.functional as F


# 경로 설정
model_weight_save_path = "DjangoProject3/resnet50_epoch_50_thisreal.pth"
num_classes = 10

# ResNet-50 모델 정의 및 로드
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)

# 모델 가중치 로드
checkpoint = torch.load(model_weight_save_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint, strict=False)
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# 이미지에서 임베딩 벡터 추출
def extract_features(image, model, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        features = model(image)
    return features


# 유클리드 거리 계산 함수
def calculate_distance(embedding1, embedding2):
    return torch.dist(embedding1, embedding2)


# 임베딩 벡터를 이용한 거리 기반 분류 함수
def classify_based_on_distance(image_embedding, class_embeddings, threshold=1.0):
    min_distance = float('inf')
    best_class = None

    for class_label, class_embedding in class_embeddings.items():
        distance = calculate_distance(image_embedding, class_embedding)
        if distance < min_distance:
            min_distance = distance
            best_class = class_label

    if min_distance > threshold:
        return "기타"
    else:
        return best_class


class ImageClassificationView(APIView):

    def post(self, request, *args, **kwargs):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']

            # 이미지 전처리
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            try:
                image = Image.open(image).convert('RGB')
                image = transform(image).unsqueeze(0).to(device)

                # 모델 예측
                with torch.no_grad():
                    outputs = model(image)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    _, predicted = torch.max(outputs, 1)
                    predicted_class_index = predicted.item()
                    confidence = probabilities[predicted_class_index].item()

                    # 클래스 레이블 설정
                    class_labels = {0: '톱', 1: '공업용가위', 2: '그라인더', 3: '니퍼', 4: '드라이버'
                        , 5: '망치', 6: '스패너', 7: '전동드릴', 8: '줄자', 9: '버니어캘리퍼스'}

                    # 필요한 레이블로 대체

                    if confidence < 0.5:
                        predicted_class_label = "기타"
                    else:
                        predicted_class_label = class_labels.get(predicted_class_index, "기타")

                    class_confidences = {class_labels[i]: round(probabilities[i].item(), 4) for i in
                                         range(len(class_labels))}

                response_data = {
                    'predicted_class_index': predicted_class_index,
                    'predicted_class_label': predicted_class_label,
                    'confidence': confidence,
                    'class_confidences': class_confidences
                }

                return Response(response_data, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)