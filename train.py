import paddle

from core.model import UNet
from solver.dateset_ import MyDataset
from solver.loss import Loss


train_dataset = MyDataset('sk1491/train/train_pair.lst')
test_dataset = MyDataset('sk1491/test/test_pair.lst')

for i in train_dataset:
    print(len(i))
    print(i[0][0].shape)
    # print(i[1][0].shape)
    # print(len(i[0]))
    break

model = paddle.Model(UNet())

# model.summary((1,3,256,256))
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
              loss=Loss())
callback = paddle.callbacks.VisualDL(log_dir='visualdl_log_dir')



model.fit(train_dataset,
          eval_data=test_dataset,
          eval_freq=1,
          epochs=100,
          batch_size=1,
          save_dir='model',
          log_freq=100,
          save_freq=10,
          callbacks=callback,
          verbose=1)
model.save('inference_model/LSTMModel', training=False)