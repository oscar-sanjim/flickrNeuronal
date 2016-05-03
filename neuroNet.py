from __future__ import division
from __future__ import print_function
from PIL import Image
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import svm
from sklearn import metrics
from StringIO import StringIO
from urlparse import urlparse
import urllib2
import sys
import os


def process_directory(directory):
    training = []
    for root, _, files in os.walk(directory):  # Method that calls the root, the directories and the files
        for file_name in files:  # Go through each file
            file_path = os.path.join(root, file_name)  # Get the absolute path
            img_feature = process_image_file(file_path)  # Get each image feature vector
            if img_feature:
                training.append(img_feature)  # Append to the training list
    return training


def process_image_file(image_path):
    image_fp = StringIO(open(image_path, 'rb').read())  # Open a file as a string
    try:
        image = Image.open(image_fp)  # Open image in the Image library
        return process_image(image)  # Get the image feature vector
    except IOError:  # Error Handling
        return None


def process_image_url(image_url):
    parsed_url = urlparse(image_url)  # Parse the string to a valid URL
    request = urllib2.Request(image_url)  # Prepare the request
    # Add headers to the request
    request.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0')
    request.add_header('Referrer', parsed_url.netloc)  # Add more headers
    net_data = StringIO(urllib2.build_opener().open(request).read())  # Get the image data
    image = Image.open(net_data)  # Open image in the Image library
    return process_image(image)  # Get the image feature vector


# Get property vector
def process_image(image, blocks=4):
    if not image.mode == 'RGB':  # Check the color mode of the image
        return None
    feature = [0] * blocks * blocks * blocks  # Array of colors, the bigger the size, the more slow and accurate
    pixel_count = 0
    for pixel in image.getdata():  # Get each pixel in the image adn divide each color to a gradient
        r_idx = int(pixel[0] / (256 / blocks))  # Calculate the red gradient in the pixel
        g_idx = int(pixel[1] / (256 / blocks))  # Calculate the green gradient in the pixel
        b_idx = int(pixel[2] / (256 / blocks))  # Calculate the blue gradient in the pixel
        # Calculate the darkness gradient based more in the blue output
        idx = r_idx + g_idx * blocks + b_idx * blocks * blocks
        feature[idx] += 1  # Add a point to the gradient of darkness
        pixel_count += 1  # Add to the pixel counter to normalize the size
    return [x / pixel_count for x in feature]  # Return the vector normalized by the size


# Train the neural network
def train(training_path_a, training_path_b, print_metrics=True):
    if not os.path.isdir(training_path_a):
        raise IOError('%s is not a directory' % training_path_a)
    if not os.path.isdir(training_path_b):
        raise IOError('%s is not a directory' % training_path_b)

    training_a = process_directory(training_path_a)  # Get the list of feature vectors from first directory
    training_b = process_directory(training_path_b)  # Get the list of feature vectors from second directory
    # Join both lists
    data = training_a + training_b
    # Create a training vector corresponding to the characteristics of each type of image
    target = [1] * len(training_a) + [0] * len(training_b)
    # split training data in a train set and a test set. The test set will
    # contain 20% of the total
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(data,
                                                                         target, test_size=0.20)
    # define the parameter search space
    parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000],
                  'gamma': [0.01, 0.001, 0.0001]}
    # search for the best classifier within the search space and return it
    clf = grid_search.GridSearchCV(svm.SVC(), parameters).fit(x_train, y_train)
    classifier = clf.best_estimator_
    if print_metrics:
        print()
        print('Parameters:', clf.best_params_)
        print()
        print('Best classifier score')
        print(metrics.classification_report(y_test,
                                            classifier.predict(x_test)))
    return classifier


def main():
    path_day = "/home/oscar/Documents/ISC/8vo/Sistemas_Inteligentes/Flickr/day"
    path_night = "/home/oscar/Documents/ISC/8vo/Sistemas_Inteligentes/Flickr/night"

    print('Training classifier...')
    classifier = train(path_day, path_night, print_metrics=False)  # Train the classifier
    while True:
        try:
            print("Input an image url (enter to exit): "),
            image_url = raw_input()  # Get an image to test
            if not image_url:
                break
            features = process_image_url(image_url)  # Process testing image
            res = classifier.predict(features)  # Predict the type of the image
            print(res)  # Print value
            if res == 0:  # Print readable version
                print("Night")
            else:
                print("Day")

        except (KeyboardInterrupt, EOFError):
            break
        except:
            exception = sys.exc_info()[0]
            print(exception)


if __name__ == '__main__':
    main()
