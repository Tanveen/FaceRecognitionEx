import cStringIO
import base64
    
# flask imports
from flask import Flask, request, request_finished, json, abort, make_response, Response, jsonify
import sys
sys.path.append("../../..")

# facenet import
from facenet import get_image_paths_and_labels, read_images_from_disk, random_rotate_image, read_and_augment_data, crop, flip
from facenet import load_data 

app = Flask(__name__)


@ThrowsWebAppException(error_code = IMAGE_DECODE_ERROR)
def read_image(base64_image):
    
    enc_data = base64.b64decode(base64_image)
    file_like = cStringIO.StringIO(enc_data)
    im = Image.open(file_like)
    im = im.convert("L")
    return im

def preprocess_image(image_data):
    image = read_image(image_data)
    return image

# Get the prediction from the global model.
@ThrowsWebAppException(error_code = PREDICTION_ERROR)
def get_prediction(image_data):
    image = preprocess_image(image_data)
    prediction = facenet.load_model(image)
    return prediction
# OR  

#def read_images_from_disk(input_queue):
#    """Consumes a single filename and label as a ' '-delimited string.
#    Args:
#      filename_and_label_tensor: A scalar string tensor.
#    Returns:
#      Two tensors: the decoded image, and the string label.
#    """
#    label = input_queue[1]
#    file_contents = tf.read_file(input_queue[0])
#    example = tf.image.decode_image(file_contents, channels=3)
#    return example, label

#def random_rotate_image(image):
#    angle = np.random.uniform(low=-10.0, high=10.0)
#    return misc.imrotate(image, angle, 'bicubic')

#def read_and_augment_data(image_list, label_list, image_size, batch_size, max_nrof_epochs, 
#        random_crop, random_flip, random_rotate, nrof_preprocess_threads, shuffle=True):
    
#    images = ops.convert_to_tensor(image_list, dtype=tf.string)
#    labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
    
#    # Makes an input queue
#    input_queue = tf.train.slice_input_producer([images, labels],
#        num_epochs=max_nrof_epochs, shuffle=shuffle)

#    images_and_labels = []
#    for _ in range(nrof_preprocess_threads):
#        image, label = read_images_from_disk(input_queue)
#        if random_rotate:
#            image = tf.py_func(random_rotate_image, [image], tf.uint8)
#        if random_crop:
#            image = tf.random_crop(image, [image_size, image_size, 3])
#        else:
#            image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
#        if random_flip:
#            image = tf.image.random_flip_left_right(image)
#        #pylint: disable=no-member
#        image.set_shape((image_size, image_size, 3))
#        image = tf.image.per_image_standardization(image)
#        images_and_labels.append([image, label])

#    image_batch, label_batch = tf.train.batch_join(
#        images_and_labels, batch_size=batch_size,
#        capacity=4 * nrof_preprocess_threads * batch_size,
#        allow_smaller_final_batch=True)
  
#    return image_batch, label_batch

@app.route('/api/recognize', methods=['GET', 'POST'])
def identify():
    if request.headers['Content-Type'] == 'application/json':
            try:
                image_data = request.json['image']
            except:
                raise WebAppException(error_code=MISSING_ARGUMENTS)
            prediction = get_prediction(image_data)
            response = jsonify(name = prediction) 
            return response
    else:
            raise WebAppException(error_code=INVALID_FORMAT)


if __name__ == '__main__':
    app.run(debug=True)
            

  
