# Data Pre-processing

The images are stored in a columnar data format, `parquet`. It is like CSV, but it is a specialized format for columnar data to achieve higher performance. `Pandas` provides a handy `read_parquet` method read this file.

```python
df = pd.read_parquet(TRAIN[0])
```

## Display images 

The following code block is about plotting an original and pre-processed images on the left and the right sides respectively. The `subplots` method creates `n_imgs` * 2 numbers of sub-plots. The number of rows and columns are indicated by the `1st` and `2nd` parameters respectively. The `figsize` parameter determines the size of the entire plot in `inch`. The size has to be specified with width and height in order.

```python
n_imgs   = 8                         
fig, axs = plt.subplots(n_imgs, 2, figsize=(10, 5*n_imgs))

for idx in range(n_imgs):
    img0 = 255 - df.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH).astype(uint8)
    img = get_crop_resize(img0)

    axs[idx,0].imshow(img0, cmap='gray')
    axs[idx,0].set_title('Original image') 
    axs[idx,0].axis('off') 
    axs[idx,1].imshow(img, cmap='gray') 
    axs[idx,1].set_title('Crop & resize') 
    axs[idx,1].axis('off') 
plt.show()
```

The `img0` is an original image while the `img` is the pre-processed one. The original data has a number of columns, and the very first column is used to indicate image ids. In such sense, it is possible to collect every columns except for the first one via `df.iloc[idx, 1:]`. That is we could retrieve every pixel values of a specific `idx`-th image.

Then `df.iloc[idx, 1:].values` convert the values stored in the dataframe columns into numpy array (the same thing could be done with `to_numpy` method). It is ok to go with this, but it has to be reshaped into 2d array if you want to leverage some CNN models. This can be done by `reshape` method with `HEIGHT` and `WIDTH` parameters. `HEIGHT` * `WIDTH` is the exact same number of the number of columns.

The reason for subtracting pixel values from `255` is that the original image is kind of being inverted. The background color is white which is near `255` or `#ffffff`value. It is possible to use it directly, but two high numbers are not so good to train the model. Instead the interesting pixels (like where the number image is) should have higher value.

<img src="./img1.png"/>

After retrieving the image pixel array from the dataframe, we can crop and resize the image. The main purpose of this task could be described with the Picture above. The hand written character in the original image is not centered, and there are plenty of marginal space which is useless for this kind of data. It is better to cut the unnecessary space and position the character part centered.

```python
def get_crop_resize(img0):
    #normalize each image by its max val
    img = (img0*(255.0/img0.max())).astype(np.uint8) 
    img = crop_resize(img) 
    return img
```

Inside the `get_crop_resize` function, the actual cropping and resizeing function is located. That is where the actual process gets happened via `crop_resize` function. `(img0*(255.0/img0.max()))` part is just normalization process. (not sure yet). Let's inspect the `crop_resize` function in more detail.

```python
def crop_resize(img0, size=SIZE, pad=16):
    img, lx, ly = cropped_img(img0)
    img = padded_img(img, lx, ly, pad)
    return cv2.resize(img,(size,size))
```

As you can see, `crop_resize` function does three things. First, make the image cropped. Second, some small background pixels are padded to to make sure every images are in the same size. For instance, if the height is longer than width, the width will be padded to expand the size. Lastly, the image gets resized to become a desired size.

```python
def cropped_img(img0, thres=80):
    ymin,ymax,xmin,xmax = bbox(img0[5:-5, 5:-5] > thres)

    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax, xmin:xmax]
    img[img < 28] = 0

    lx = xmax - xmin
    ly = ymax - ymin

    return img, lx, ly
```

Let's look into the `cropped_img` function. The `bbox` function will be explained shortly, but the returned values are the indicies where the pixel value is above the `thres`. Since there could be some noisy values, the `thres` is set to `80` here. So, `ymin` and `ymax` are the pixel location of bottom and top respectively where the value exceeds 80. Likewise, `xmin` and `xmax` are the pixel location of left and right.

The four lines of code with `+/-13` and `+/-10` things are attempts to give some extra space so that the charater of the image can be centered. It will prevent the character from being located at the exact corners.

`img[img < 28] = 0` erase some noisy pixel values. Finally, `lx` and `ly` is the size of width and height. The same thing could be achieved via `img.shape` I guess. These values will be used to determine which size should be padded and how much in `padded_img` function below.

```python
def padded_img(img, lx, ly, pad):
    l = max(lx,ly) + pad
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    return img
```

`padded_img` function performs padding. First, `max(ly, ly)` deteremines wheter width or height should be padded more. `+pad` gives more space additionally. You can get some sense what np.pad does through the following exmaple.

```
>>> a = [[1, 2], [3, 4]]
>>> np.pad(a, ((3, ), (2, )), mode='constant')
array([[0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 1, 2, 0, 0],
       [0, 0, 3, 4, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0])
```

The original values were `[[1,2],[3,4]]`, and you could find it on the center of the resuling array. As you can see, three rows are added to the top and bottom, and two columns are added to the left and right to surroung the original values.

In this sense, `np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')` pads evenly to the every directions (note `//2` operation). When mode is `constant` the padded values are `0`. 


```python
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax
```

The last function we will look at is `bbox`. Remeber this function was used in the first line of the `crop_resize` function. The input parameter `img` is not an actual image, but it is assumed to be an boolean array with the same size to the original image. In such array, we could find easily where the pixels of our interests are located, and that is done with `np.where` functionality.  

`np.where` returns the indices of array indicating the location where the `True` values are. `np.where(rows)[0]`'s [0] is just to access the array. `[[0, -1]]` thing will give you the first and the last indices from the array. So you now know the location of the starting pixel for the character.

### References
- [Apache Parquet](http://parquet.apache.org/)
- [Pandas pd.read_parquet](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_parquet.html)
- [Pandas Series values property](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.values.html)
- [np.pad](https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html)
- [np.any](https://docs.scipy.org/doc/numpy/reference/generated/numpy.any.html)
- [np.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)

# Modelling
