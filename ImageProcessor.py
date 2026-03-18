import cv2
import numpy as np
# ═══════════════════════════════════════════════════════════════
#  IMAGE PROCESSOR
# ═══════════════════════════════════════════════════════════════
class ImageProcessor:
    @staticmethod
    def apply(img, mode, params, custom_kernel=None):
        if mode == "None": return img
        elif mode == "Add (+)":
            return np.clip(img.astype(np.int16)+params.get("Value",50),0,255).astype(np.uint8)
        elif mode == "Subtract (-)":
            return np.clip(img.astype(np.int16)-params.get("Value",50),0,255).astype(np.uint8)
        elif mode == "Multiply (×)":
            f=params.get("Factor (x10)",10)/10.0
            return np.clip(img.astype(np.float32)*f,0,255).astype(np.uint8)
        elif mode == "Divide (÷)":
            f=max(0.1,params.get("Factor (x10)",10)/10.0)
            return np.clip(img.astype(np.float32)/f,0,255).astype(np.uint8)
        elif mode == "Histogram Equalization":
            yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
            yuv[:,:,0]=cv2.equalizeHist(yuv[:,:,0])
            return cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
        elif mode == "CLAHE":
            clip=params.get("Clip Limit (x10)",20)/10.0
            grid=params.get("Grid Size",8)
            clahe=cv2.createCLAHE(clipLimit=clip,tileGridSize=(grid,grid))
            yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
            yuv[:,:,0]=clahe.apply(yuv[:,:,0])
            return cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
        elif mode == "Gamma Correction":
            g=params.get("Gamma (x10)",10)/10.0
            table=np.array([((i/255.0)**(1.0/g))*255 for i in range(256)]).astype("uint8")
            return cv2.LUT(img,table)
        elif mode == "Salt and Pepper":
            res=img.copy(); prob=params.get("Noise Prob (%)",5)/100.0
            rnd=np.random.rand(*img.shape[:2])
            res[rnd<prob/2]=0; res[rnd>1-prob/2]=255
            return res
        elif mode == "Gaussian Noise":
            sigma=params.get("Sigma",20)
            gauss=np.random.normal(0,sigma,img.shape).astype(np.float32)
            return np.clip(img.astype(np.float32)+gauss,0,255).astype(np.uint8)
        elif mode == "Median Filter":
            k=params.get("Kernel Size (Odd)",3); k=k if k%2 else k+1
            return cv2.medianBlur(img,k)
        elif mode == "Gaussian Blur":
            k=params.get("Kernel Size (Odd)",5); k=k if k%2 else k+1
            return cv2.GaussianBlur(img,(k,k),0)
        elif mode == "Bilateral Filter":
            d=params.get("Diameter",9); sc=params.get("Sigma Color",75)
            return cv2.bilateralFilter(img,d,sc,sc)
        elif mode == "Sharpen":
            s=params.get("Strength (x10)",10)/10.0
            k=np.array([[0,-1,0],[-1,4+s,-1],[0,-1,0]],dtype=np.float32)
            return np.clip(cv2.filter2D(img,-1,k),0,255).astype(np.uint8)
        elif mode == "Emboss":
            k=np.array([[-2,-1,0],[-1,1,1],[0,1,2]],dtype=np.float32)
            return np.clip(cv2.filter2D(img,-1,k)+128,0,255).astype(np.uint8)
        elif mode == "Laplacian":
            blend=params.get("Blend (%)",50)/100.0
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            lap=cv2.convertScaleAbs(cv2.Laplacian(gray,cv2.CV_64F))
            return cv2.addWeighted(img,1-blend,cv2.cvtColor(lap,cv2.COLOR_GRAY2BGR),blend,0)
        elif mode == "Sobel Edges":
            k=params.get("Kernel Size",3); k=k if k%2 else k+1
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=k)
            gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=k)
            return cv2.cvtColor(cv2.convertScaleAbs(np.sqrt(gx**2+gy**2)),cv2.COLOR_GRAY2BGR)
        elif mode == "Canny Edges":
            lo=params.get("Min Threshold",100); hi=params.get("Max Threshold",200)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(cv2.Canny(gray,lo,hi),cv2.COLOR_GRAY2BGR)
        elif mode == "Erosion":
            return cv2.erode(img,np.ones((5,5),np.uint8),iterations=params.get("Iterations",1))
        elif mode == "Dilation":
            return cv2.dilate(img,np.ones((5,5),np.uint8),iterations=params.get("Iterations",1))
        elif mode == "ORB Features":
            orb=cv2.ORB_create(nfeatures=params.get("Max Features",500))
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            kp,_=orb.detectAndCompute(gray,None)
            return cv2.drawKeypoints(img,kp,None,color=(0,255,0),flags=0)
        elif mode == "Low Pass Filter":
            k=params.get("Kernel Size (Odd)",5); k=k if k%2 else k+1
            return cv2.GaussianBlur(img,(k,k),0)
        elif mode == "High Pass Filter":
            s=params.get("Strength (x10)",10)/10.0
            k=np.array([[0,-1,0],[-1,4+s,-1],[0,-1,0]],dtype=np.float32)
            return np.clip(cv2.filter2D(img,-1,k),0,255).astype(np.uint8)
        elif mode == "SIFT Features":
            try:
                sift=cv2.SIFT_create(nfeatures=params.get("Max Features",500))
            except Exception:
                sift=cv2.xfeatures2d.SIFT_create(nfeatures=params.get("Max Features",500)) if hasattr(cv2,'xfeatures2d') else None
            if sift is None: return img
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            kp,_=sift.detectAndCompute(gray,None)
            out=img.copy()
            cv2.drawKeypoints(out,kp,out,color=(255,100,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.putText(out,f"SIFT KP: {len(kp)}",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,100,0),2)
            return out
        elif mode == "Segmentation (Otsu)":
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            _,th=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            return cv2.cvtColor(th,cv2.COLOR_GRAY2BGR)
        elif mode == "Custom Kernel":
            if custom_kernel is not None and custom_kernel.size>0:
                return np.clip(cv2.filter2D(img,-1,custom_kernel),0,255).astype(np.uint8)
        return img


