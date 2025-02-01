# Test set

 **Images** <br>
 Please download the images from VisDrone [test-dev](https://github.com/VisDrone/VisDrone-Dataset). <br>
 
 **Annotations** <br>
 The annotations is located in [here](https://github.com/sunzc-sunny/refdrone/blob/main/dataset/RefDrone_test_mdetr.json). <br>
 Annotation is mdetr style with keys: "images", "annotaions". <br>

#### Images
Contains information about each image in the dataset:

| Key | Description | Example |
|-----|-------------|-------------|
| `file_name` | Name of the image file | "0000189_00297_d_0000198.jpg" |
| `height` | Height of the image in pixels | 540 |
| `width` | Width of the image in pixels | 960 |
| `id` | Unique identifier for the image | 0 |
| `caption` | Text description of the image | "The black cars on the road" |
| `dataset_name` | Name of the dataset | "RefDrone" |
| `token_negative` | List of tokens not associated with target objects | [[0, 3], [4, 9]] |

#### Annotations
Contains information about the annotations for each image:

| Key | Description | Example |
|-----|-------------|-------------|
| `area` | Area of the annotated region | 1584 |
| `iscrowd` | if the annotation represents a crowd | 0|
| `image_id` | ID of the image this annotation belongs to | 0 |
| `category_id` | Category identifier for the annotated object | "5" |
| `id` | Unique identifier for the annotation | 0 |
| `empty` | if the annotation is empty | 0 |
| `bbox` | Bounding box coordinates [x, y, width, height] | [100, 200, 10, 20] |
| `tokens_positive` | List of tokens associated with the target object | [[2,4]] |


 
 
