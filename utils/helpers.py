import numpy as np
import torch

IMG_SCALE  = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

def maybe_download(model_name, model_url, model_dir=None, map_location=None):
    import os, sys
    from six.moves import urllib
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = '{}.pth.tar'.format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urllib.request.urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)

def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD
import visdom
import time
import numpy as np
#import unicode
class Visualizer(object):
   '''
   封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
   或者`self.function`调用原生的visdom接口
   比如 
   self.text('hello visdom')
   self.histogram(t.randn(1000))
   self.line(t.arange(0, 10),t.arange(1, 11))
   '''

   def __init__(self, env='default', **kwargs):
       self.vis = visdom.Visdom(env=env, **kwargs)
       
       # 画的第几个数，相当于横坐标
       # 比如（’loss',23） 即loss的第23个点
       self.index = {} 
       self.log_text = ''
   def reinit(self, env='default', **kwargs):
       '''
       修改visdom的配置
       '''
       self.vis = visdom.Visdom(env=env, **kwargs)
       return self

   def plot_many(self, d):
       '''
       一次plot多个
       @params d: dict (name, value) i.e. ('loss', 0.11)
       '''
       for k, v in d.iteritems():
           self.plot(k, v)

   def img_many(self, d):
       for k, v in d.iteritems():
           self.img(k, v)

   def plot(self, name, y, **kwargs):
       '''
       self.plot('loss', 1.00)
       '''
       x = self.index.get(name, 0)
       self.vis.line(Y=np.array([y]), X=np.array([x]),
                     win=str(name),
                     opts=dict(title=name),
                     update=None if x == 0 else 'append',
                     **kwargs
                     )
       self.index[name] = x + 1

   def img(self, name, img_, **kwargs):
       '''
       self.img('input_img', t.Tensor(64, 64))
       self.img('input_imgs', t.Tensor(3, 64, 64))
       self.img('input_imgs', t.Tensor(100, 1, 64, 64))
       self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
       '''
       self.vis.images(img_.cpu().numpy(),
                      win=unicode(name),
                      opts=dict(title=name),
                      **kwargs
                      )

   def log(self, info, win='log_text'):
       '''
       self.log({'loss':1, 'lr':0.0001})
       '''

       self.log_text += ('[{time}] {info} <br>'.format(
                           time=time.strftime('%m%d_%H%M%S'),\
                           info=info)) 
       self.vis.text(self.log_text, win)   

   def __getattr__(self, name):
       '''
       self.function 等价于self.vis.function
       自定义的plot,image,log,plot_many等除外
       '''
       return getattr(self.vis, name)