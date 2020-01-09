```python
df = pd.read_parquet(TRAIN[0]) 
```

```python
n_imgs   = 8                         
fig, axs = plt.subplots(n_imgs, 2, figsize=(10, 5*n_imgs))

for idx in range(n_imgs):
    img = get_crop_resize(df, idx)

    axs[idx,0].imshow(img0, cmap='gray')
    axs[idx,0].set_title('Original image') 
    axs[idx,0].axis('off') 
    axs[idx,1].imshow(img, cmap='gray') 
    axs[idx,1].set_title('Crop & resize') 
    axs[idx,1].axis('off') 
plt.show()
```

<img src="./img1.png"/>

```python
def get_crop_resize(df, idx):
    img0 = df.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH)
    img0 = 255 - img0.astype(np.uint8)

    #normalize each image by its max val
    img = (img0*(255.0/img0.max())).astype(np.uint8) 
    img = crop_resize(img) 
    
    return img
```

```python
def crop_resize(img0, size=SIZE, pad=16):
    img, lx, ly = cropped_img(img0)
    img = padded_img(img, lx, ly, pad)
    return cv2.resize(img,(size,size))
```

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

```python
def padded_img(img, lx, ly, pad):
    l = max(lx,ly) + pad
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    return img
```

```python
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax
```