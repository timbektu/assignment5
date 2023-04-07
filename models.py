import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        #TODO: how does nn.linear(3,3) take input(B,n1,3) and also input(B,n2,n1,3)?
        self.input_transform1 = nn.Linear(3,3, bias=False) #TODO: maybe add bias so that translation is taken care of?
        self.mlp1_1 = nn.Linear(3,64)
        self.bn1_1 = nn.BatchNorm1d(64)
        self.mlp1_2 = nn.Linear(64, 64)
        self.bn1_2 = nn.BatchNorm1d(64)
        self.feat_transform = nn.Linear(64,64, bias = False)
        
        self.mlp2_1 = nn.Linear(64,64)
        self.bn2_1 = nn.BatchNorm1d(64)
        self.mlp2_2 = nn.Linear(64,128)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.mlp2_3 = nn.Linear(128,1024)
        self.bn2_3 = nn.BatchNorm1d(1024)

        #TODO: maxpool?

        self.mlp3_1 = nn.Linear(1024,512)
        self.bn3_1 = nn.BatchNorm1d(512)
        self.mlp3_2 = nn.Linear(512,256)
        self.bn3_2 = nn.BatchNorm1d(256)
        self.mlp3_3 = nn.Linear(256,num_classes)

        self.relu = nn.ReLU()


    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''

        points = self.input_transform1(points)
        points = self.mlp1_1(points)
        points = self.relu(self.bn1_1(points.transpose(1,2)))
        # pdb.set_trace()
        points = self.mlp1_2(points.transpose(1,2))
        points = self.relu(self.bn1_2(points.transpose(1,2)))
        # points = self.relu(points)

        points = self.feat_transform(points.transpose(1,2))
        # pdb.set_trace()
        points = self.mlp2_1(points)
        points = self.relu(self.bn2_1(points.transpose(1,2)))
        # points = self.relu(points)
        points = self.mlp2_2(points.transpose(1,2))
        points = self.relu(self.bn2_2(points.transpose(1,2)))
        # points = self.relu(points)
        points = self.mlp2_3(points.transpose(1,2))
        points = self.relu(self.bn2_3(points.transpose(1,2)))
        # points = self.relu(points)

        # pdb.set_trace()
        global_feat,_ = torch.max(points, dim=-1) #TODO: should be B x 1024

        scores = self.mlp3_1(global_feat)
        # points = self.relu(self.bn3_1(points))
        points = self.relu(points)

        scores = self.mlp3_2(scores)
        # points = self.relu(self.bn3_2(points))
        points = self.relu(points)

        scores = self.mlp3_3(scores)
        # points = self.relu(self.bn3_3(points))
        # points = self.relu(points)
        
        # scores = F.softmax(scores, dim =-1)
        # pred_probs, pred_labels =  torch.max(scores, dim=-1)

        return scores

# class cls_model(nn.Module):
#     def __init__(self, num_classes=3):
#         super(cls_model, self).__init__()
#         self.conv1 = nn.Conv1d(3, 64, 1) # Lin=Lout=N
#         self.conv2 = nn.Conv1d(64, 128, 1)
#         self.conv3 = nn.Conv1d(128, 1024, 1)
        
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)

#         self.f1 = nn.Linear(1024, 512)
#         self.f2 = nn.Linear(512, 256)
#         self.f3 = nn.Linear(256, num_classes)

#         self.relu = nn.ReLU()

#     def forward(self, points):
#         '''
#         points: tensor of size (B, N, 3)
#                 , where B is batch size and N is the number of points per object (N=10000 by default)
#         output: tensor of size (B, num_classes)
#         '''
#         # shared weights MLP
#         points = torch.transpose(points, 1, 2) # B, 3, N
#         out = self.relu(self.bn1(self.conv1(points))) # B, 64, N
#         out = self.relu(self.bn2(self.conv2(out))) # B, 128, N
#         out = self.relu(self.bn3(self.conv3(out))) # B, 1024, N
        
#         # max pool
#         out, _ = torch.max(out, dim=-1) # global feature, B x 1024

#         # MLP
#         # out = self.relu(self.f1(out))
#         # out = self.relu(self.f2(out))
#         out = self.f1(out)
#         out = self.f2(out)
#         out = nn.functional.softmax(self.f3(out), -1) # B x k
#         return out


# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()

        #TODO: skipped transform layer?
        self.conv1_1 = nn.Conv1d(3, 64, 1)
        self.conv1_2 = nn.Conv1d(64,64, 1)

        self.feat_transform = nn.Linear(64,64) #TODO: bias is true here?

        self.conv2_1 = nn.Conv1d(64,64,1)
        self.conv2_2 = nn.Conv1d(64, 128, 1)
        self.conv2_3 = nn.Conv1d(128, 1024, 1)

        self.conv3_1 = nn.Conv1d(1088,512,1)
        self.conv3_2 = nn.Conv1d(512,256, 1)
        self.conv3_3 = nn.Conv1d(256,128,1)

        self.conv4_1 = nn.Conv1d(128,128,1)
        self.conv4_2 = nn.Conv1d(128,num_seg_classes,1)

        self.relu = nn.ReLU()

        self.bn1_1 = nn.BatchNorm1d(64)
        self.bn1_2 = nn.BatchNorm1d(64)

        self.bn2_1 = nn.BatchNorm1d(64)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.bn2_3 = nn.BatchNorm1d(1024)

        self.bn3_1 = nn.BatchNorm1d(512)
        self.bn3_2 = nn.BatchNorm1d(256)
        self.bn3_3 = nn.BatchNorm1d(128)

        # self.bn4_1 = nn.BatchNorm1d(128)s



    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        points = points.transpose(1,2)
        points = self.relu(self.bn1_1(self.conv1_1(points)))
        points = self.relu(self.bn1_2(self.conv1_2(points)))

        points = self.feat_transform(points.transpose(1,2)).transpose(1,2)
        points_skip = points.clone()

        points = self.relu(self.bn2_1(self.conv2_1(points)))
        points = self.relu(self.bn2_2(self.conv2_2(points)))
        points = self.relu(self.bn2_3(self.conv2_3(points)))

        global_feat, _ = torch.max(points, dim=-1) 
        global_feat = global_feat.unsqueeze(-1).repeat(1,1,points_skip.shape[-1])
        
        seg_feat = torch.hstack([points_skip, global_feat])
        
        seg_feat = self.relu(self.bn3_1(self.conv3_1(seg_feat)))
        seg_feat = self.relu(self.bn3_2(self.conv3_2(seg_feat)))
        seg_feat = self.relu(self.bn3_3(self.conv3_3(seg_feat)))

        seg_feat = self.relu(self.conv4_1(seg_feat))
        seg_feat = self.conv4_2(seg_feat)
        # combined_feats = 

        scores = seg_feat.transpose(1,2)
        # scores = F.softmax(scores, dim =-1)

        return scores




