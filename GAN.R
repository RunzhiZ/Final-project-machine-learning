library(keras)
setwd('/blue/datta/runzhi.zhang/Machine learning/')
latent_dim <- 28
height <- 28
width <- 28
channels <- 1
generator_input <- layer_input(shape = c(latent_dim))
generator_output <- generator_input %>%
  layer_dense(units = 128*7*7) %>%                             
  layer_activation_leaky_relu() %>%                                  
  layer_reshape(target_shape = c(7, 7, 128)) %>%
  layer_upsampling_2d(size = c(2, 2)) %>%
  layer_conv_2d(filters = 256, kernel_size = c(5,5),
                padding = "same") %>%
  layer_activation_leaky_relu() %>%
  layer_upsampling_2d(size = c(2, 2)) %>%
  layer_conv_2d_transpose(filters = 128, kernel_size = c(5,5),            
                          padding = "same") %>%         
  layer_activation_leaky_relu() %>%                                  
  layer_conv_2d(filters = 128, kernel_size = c(5,5),
                padding = "same") %>%
  layer_activation_leaky_relu() %>%
  layer_conv_2d(filters = 128, kernel_size = c(5,5),
                padding = "same") %>%
  layer_activation_leaky_relu() %>%
  layer_conv_2d(filters = channels, kernel_size = c(2,2),                 
                activation = "sigmoid", padding = "same")               
generator <- keras_model(generator_input, generator_output)          

discriminator_input <- layer_input(shape = c(height, width, channels))
discriminator_output <- discriminator_input %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),padding = 'same', strides = c(2,2)) %>%
  layer_activation_leaky_relu() %>%
  layer_dropout(0.3) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3),padding = 'same', strides = c(1,1)) %>%
  layer_activation_leaky_relu() %>%
  layer_dropout(0.3) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3),padding = 'same', strides = c(2,2)) %>%
  layer_activation_leaky_relu() %>%
  layer_dropout(0.3) %>%
  layer_conv_2d(filters = 256, kernel_size = c(3,3),padding = 'same', strides = c(1,1)) %>%
  layer_activation_leaky_relu() %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.4) %>%                                            
  layer_dense(units = 1, activation = "sigmoid")                           
discriminator <- keras_model(discriminator_input, discriminator_output)    
discriminator_optimizer <- optimizer_rmsprop(
  lr = 0.0008,
  clipvalue = 1.0,                                                         
  decay = 1e-8                                                             
)
discriminator %>% compile(
  optimizer = discriminator_optimizer,
  loss = "binary_crossentropy"
)

freeze_weights(discriminator)                         
gan_input <- layer_input(shape = c(latent_dim))
gan_output <- discriminator(generator(gan_input))
gan <- keras_model(gan_input, gan_output)
gan_optimizer <- optimizer_rmsprop(
  lr = 0.0004,
  clipvalue = 1.0,
  decay = 1e-8
)
gan %>% compile(
  optimizer = gan_optimizer,
  loss = "binary_crossentropy"
)

mnist <- dataset_mnist()
c(c(x_train, y_train), c(x_test, y_test)) %<-% mnist
x_train <- x_train / 255
x_train <- array_reshape(x_train, dim =c(dim(x_train), 1))
x_test <- x_test / 255
x_test <- array_reshape(x_test, dim =c(dim(x_test), 1))


iterations <- 1000
batch_size <- 20                                                
start <- 1
for (step in 1:iterations) {
  random_latent_vectors <- matrix(rnorm(batch_size * latent_dim),         
                                  nrow = batch_size, ncol = latent_dim)
  generated_images <- generator %>% predict(random_latent_vectors)    
  stop <- start + batch_size - 1                                          
  real_images <- x_train[start:stop,,,1]  
  real_images <- array_reshape(real_images,dim=c(dim(real_images),1))
  rows <- nrow(real_images)                                               
  combined_images <- array(0, dim = c(rows * 2, dim(real_images)[-1]))    
  combined_images[1:rows,,,] <- generated_images                          
  combined_images[(rows+1):(rows*2),,,] <- real_images                    
  labels <- rbind(matrix(1, nrow = batch_size, ncol = 1),                 
                  matrix(0, nrow = batch_size, ncol = 1))                 
  labels <- labels + (0.5 * array(runif(prod(dim(labels))),               
                                  dim = dim(labels)))                     
  d_loss <- discriminator %>% train_on_batch(combined_images, labels)     
  random_latent_vectors <- matrix(rnorm(batch_size * latent_dim),         
                                  nrow = batch_size, ncol = latent_dim)   
  misleading_targets <- array(0, dim = c(batch_size, 1))                  
  a_loss <- gan %>% train_on_batch(random_latent_vectors,misleading_targets)                                                                       
  start <- start + batch_size
  if (start > (nrow(x_train) - batch_size))
    start <- 1
  if (step %% 100 == 0) {                                                 
    #save_model_weights_hdf5(gan, "gan.h5")                                
    cat("discriminator loss:", d_loss, "\n")                              
    cat("adversarial loss:", a_loss, "\n")  
    }
    if(step == 10){
      generated_images_10<-generated_images
    }
    if(step == 50){
      generated_images_50<-generated_images
    }
    if(step == 200){
      generated_images_200<-generated_images
    }
    if(step == 1000){
      generated_images_1000<-generated_images
    }
    
  #   image_array_save(generated_images[1,,,] * 255,path = file.path(save_dir, paste0("generated_frog", step, ".png")))                                                                     
  #   image_array_save(real_images[1,,,] * 255,path = file.path(save_dir, paste0("real_frog", step, ".png")))                                                                     
}

save.image('GAN_result.RData')

#### 1000 ####
img_1000<-NULL
img_c<-NULL
for(i in 1:4){
  for(j in 1:5){
    img_c<-rbind(img_c,generated_images_1000[j+5*(i-1),,,])
  }
  img_1000<-cbind(img_1000,img_c)
  img_c<-NULL
}

png(file="GAN_1000.png")
plot(as.raster(img_1000))
dev.off()

#### 200 ####
img_200<-NULL
img_c<-NULL
for(i in 1:4){
  for(j in 1:5){
    img_c<-rbind(img_c,generated_images_200[j+5*(i-1),,,])
  }
  img_200<-cbind(img_200,img_c)
  img_c<-NULL
}

png(file="GAN_200.png")
plot(as.raster(img_200))
dev.off()

#### 50 ####
img_50<-NULL
img_c<-NULL
for(i in 1:4){
  for(j in 1:5){
    img_c<-rbind(img_c,generated_images_50[j+5*(i-1),,,])
  }
  img_50<-cbind(img_50,img_c)
  img_c<-NULL
}

png(file="GAN_50.png")
plot(as.raster(img_50))
dev.off()

#### 10 ####
img_10<-NULL
img_c<-NULL
for(i in 1:4){
  for(j in 1:5){
    img_c<-rbind(img_c,generated_images_10[j+5*(i-1),,,])
  }
  img_10<-cbind(img_10,img_c)
  img_c<-NULL
}

png(file="GAN_10.png")
plot(as.raster(img_10))
dev.off()



