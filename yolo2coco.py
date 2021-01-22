import json
import os
import cv2


def yolo2coco(image_dir_path, label_dir_path,save_file_name , is_normalized):

    total = {}
    # add Class Names
    class_Names =['mask', 'no-mask']
    count_ID = 1

    info = {
            'description' : '',
            'url' : '',
            'version' : '',
            'year' : 2020,
            'contributor' : '',
            'data_created' : '2020-04-14 01:45:18.567988'
            }
    total['info'] = info

    
    licenses_list = []
    licenses_0= {
            'id' : '1',
            'name' : 'your_name',
            'url' : 'your_name'
            }
    licenses_list.append(licenses_0)


    total['licenses'] = licenses_list

    # make categories
    category_list = []
    infoCoco = {'id':None,'name':None,'supercategory': 'None'}


    for elem in class_Names:
        infoCoco['id'] = count_ID
        infoCoco['name'] = elem
        category_list.append(infoCoco.copy())
        count_ID +=1
    
    
   
    total['categories'] = category_list

    
    image_list = os.listdir(image_dir_path)
    print('image length : ', len(image_list))
    label_list = os.listdir(label_dir_path)
    print('label length : ',len(label_list))
    print('Converting.........')
    image_dict_list = []
    count = 0
    for image_name in image_list :
        
        img = cv2.imread(image_dir_path+image_name)
        # print(img.shape[1])
        
        image_dict = {
                'id' : count,
                'file_name' : image_name,
                'width' : img.shape[1],
                'height' : img.shape[0],
                'date_captured' : '2020-04-14 -1:45:18.567975',
                'license' : 1, 
                'coco_url' : '',
                'flickr_url' : ''
                }
        image_dict_list.append(image_dict)
        count += 1
    total['images'] = image_dict_list

    
    label_dict_list = []
    image_count = 0
    label_count = 0
    for image_name in image_list :
        img = cv2.imread(image_dir_path+image_name)
        label = open(label_dir_path+image_name[0:-4] + '.txt','r')
        if not os.path.isfile(label_dir_path + image_name[0:-4] + '.txt'): # debug code
            print('there is no label match with ',image_dir_path + image_name)
            return
        while True:
            line = label.readline()
            if not line:
                break
            class_number, center_x,center_y,box_width,box_height = line.split()
            
            if is_normalized :
                print('normal')
                center_x =  float(center_x) * int(img.shape[1])
                center_y = float(center_y) * int(img.shape[0])
                box_width = float(box_width) * int(img.shape[1])
                box_height = float(box_height) * int(img.shape[0])
                top_left_x = center_x - int(box_width/2)
                top_left_y = center_y - int(box_height/2)

            if not is_normalized :
                center_x =  float(center_x) * int(img.shape[1])
                center_y = float(center_y) * int(img.shape[0])
                box_width = float(box_width) * int(img.shape[1])
                box_height = float(box_height) * int(img.shape[0])
                top_left_x = center_x - int(box_width/2)
                top_left_y = center_y - int(box_height/2)
                

            bbox_dict = []
            bbox_dict.append(top_left_x)
            bbox_dict.append(top_left_y)
            bbox_dict.append(box_width)
            bbox_dict.append(box_height)

            
            segmentation_list_list = []
            segmentation_list= []
            segmentation_list.append(bbox_dict[0])
            segmentation_list.append(bbox_dict[1])
            segmentation_list.append(bbox_dict[0] + bbox_dict[2])
            segmentation_list.append(bbox_dict[1])
            segmentation_list.append(bbox_dict[0]+bbox_dict[2])
            segmentation_list.append(bbox_dict[1]+bbox_dict[3])
            segmentation_list.append(bbox_dict[0])
            segmentation_list.append(bbox_dict[1] + bbox_dict[3])
            segmentation_list_list.append(segmentation_list)

            label_dict = {
                    'id' : label_count,
                    'image_id' : image_count,
                    'category_id' : int(class_number)+1,
                    'iscrowd' : 0,
                    'area' : int(bbox_dict[2] * bbox_dict[3]),
                    'bbox' : bbox_dict,
                    'segmentation' : segmentation_list_list
                    }
            label_dict_list.append(label_dict)
            label_count += 1
        label.close()
        image_count += 1

    total['annotations'] = label_dict_list

    with open(save_file_name,'w',encoding='utf-8') as make_file :
        json.dump(total,make_file, ensure_ascii=False,indent='\t')

if __name__ == '__main__':
    image_dir_path = 'img/'
    label_dir_path = 'lbl/'
    save_file_name = 'instances_ImagesTrain.json'
    is_normalized = False
    
    yolo2coco(image_dir_path, label_dir_path, save_file_name,is_normalized)