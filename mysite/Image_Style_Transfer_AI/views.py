import io, time, threading
from django.contrib.auth.hashers import make_password, check_password
from django.db import connection
from django.http import HttpResponseRedirect, HttpResponse
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core import serializers
from django.views.generic import TemplateView
from Image_Style_Transfer_AI.forms import ImageUploadForm, LoginForm, SignUpForm
from django.forms.utils import ErrorList
from django.shortcuts import render
from Image_Style_Transfer_AI.models import Art, User
from IPython.display import  display
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import vgg16

#called to produced the art and adds the result to the db
def create_art(art_db_instance, num_of_iterations):
    DEBUG = False
    image_one_path = art_db_instance.image_one.path
    content_image = np.float32(PIL.Image.open(image_one_path))
    image_two_path = art_db_instance.image_two.path
    style_image = np.float32(PIL.Image.open(image_two_path))
    content_layer_ids = [4]
    style_layer_ids = list(range(13))

    if not DEBUG:
        img = style_transfer(content_image=content_image,
                             style_image=style_image,
                             content_layer_ids=content_layer_ids,
                             style_layer_ids=style_layer_ids,
                             weight_content=1.5,
                             weight_style=10.0,
                             weight_denoise=0.3,
                             num_iterations=num_of_iterations,
                             step_size=10.0)
    else:
        time.sleep(10.0)
        print("TEST " + str(num_of_iterations))
        img = content_image

    image = np.clip(img, 0.0, 255.0)
    image = image.astype(np.uint8)
    image = PIL.Image.fromarray(image)
    image_io = io.BytesIO()
    image.save(image_io, format='JPEG')
    image_name = "result.jpg"
    image_file = InMemoryUploadedFile(image_io, None, image_name, 'image/jpeg', image_io.tell, None)
    art_db_instance.image_output.save(image_name, image_file)
    connection.close()

#called to logout
def logout(request):
    request.session.flush()
    return HttpResponseRedirect("/")

#Views
def load_index_page(request, form=ImageUploadForm()):
    if "user_email" in request.session:
        user = User.objects.filter(email=request.session["user_email"])
        if (user.count() > 0):
            user = user[:1].get()
            art = Art.objects.filter(artist=user)
            return render(request, "Image_Style_Transfer_AI/index.html", {
                "form": form,
                "art": art,
            })
    return logout(request)

#renders the login page, used by the HomeView class
def load_login_page(request, login_form=LoginForm(), signup_form=SignUpForm()):
    return render(request, "Image_Style_Transfer_AI/login.html", {
        "login_form": login_form,
        "signup_form": signup_form,
    })

#returns all art in json format for the logged in user
def json(request):
    if type(request.session["user_email"]) is not None:
        user = User.objects.filter(email=request.session["user_email"])
        if (user.count() > 0):
            user = user[:1].get()
            data = list(Art.objects.filter(artist=user))
            qs_json = serializers.serialize('json', data)
            return HttpResponse(qs_json, content_type='application/json')
    return HttpResponseRedirect("/")


#View class to handle post and get request
class HomeView(TemplateView):
    def get(self, request):
        if "user_email" in request.session:
            #return art form
            return load_index_page(request=request)
        else:
            #return login form
            return load_login_page(request=request)

    def post(self, request):
        if "user_email" in request.session:
            #creating art
            form = ImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                instance = Art()
                instance.artist = User.objects.filter(email=request.session["user_email"])[:1].get()
                instance.image_one = form.cleaned_data['image_one']
                instance.image_two = form.cleaned_data['image_two']
                instance.save()
                art_generator = threading.Thread(target=create_art, args=[instance, form.cleaned_data['num_of_iterations']])
                art_generator.setDaemon(True)
                art_generator.start()
                return HttpResponseRedirect("/")
            return load_index_page(request=request, form=form)

        else:
            # logging in
            if "login" in request.POST:
                form = LoginForm(request.POST)
            elif "signup" in request.POST:
                form = SignUpForm(request.POST)

            if form.is_valid():
                verified = False
                email = form.cleaned_data["email"]
                password = form.cleaned_data["password"]
                errors = form._errors.setdefault("myfield", ErrorList())
                #returning user
                if "login" in request.POST:
                    user_logging_in = User.objects.filter(email=email)
                    if user_logging_in.count() > 0 and check_password(password, user_logging_in[:1].get().password):
                        #details match
                        verified = True;
                    else:
                        errors.append(u"Your Email or/and passwords do not match")
                #new user
                elif "signup" in request.POST:
                    #check is passwords match
                    errors = form._errors.setdefault("myfield", ErrorList())
                    signup_error = False;
                    if User.objects.filter(email=email).count() > 0:
                        errors.append(u"Email already used.")
                        signup_error = True;
                    if password != form.cleaned_data["password_retyped"]:
                        errors.append(u"Your passwords do not match")
                        signup_error = True;
                    if not signup_error:
                        new_user = User()
                        new_user.email = email
                        new_user.password = make_password(password)
                        new_user.save()
                        verified = True

                if verified:
                    #login success
                    request.session["user_email"] = email
                    return load_index_page(request=request)

            #login failed, show errors
            if "login" in request.POST:
                return load_login_page(request=request, login_form=form)
            elif "signup" in request.POST:
                return load_login_page(request=request, signup_form=form)
            return load_login_page(request=request)


# the following code is based of https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/15_Style_Transfer.ipynb
def style_transfer(content_image, style_image, content_layer_ids, style_layer_ids, weight_content, weight_style, weight_denoise, num_iterations, step_size):
    model = vgg16.VGG16()
    session = tf.InteractiveSession(graph=model.graph)

    # generate content loss
    content_layers = model.get_layer_tensors(content_layer_ids)
    content_values = session.run(content_layers, feed_dict=model.create_feed_dict(image=content_image))
    with model.graph.as_default():
        content_layer_losses = []
        for value, layer in zip(content_values, content_layers):
            content_layer_losses.append(tf.reduce_mean(tf.square(layer - tf.constant(value))))
        loss_content = tf.reduce_mean(content_layer_losses)

    # generate style loss
    with model.graph.as_default():
        style_gram_layers = []
        for tensor in model.get_layer_tensors(style_layer_ids):
            matrix = tf.reshape(tensor, shape=[-1, int(tensor.get_shape()[3])])
            style_gram_layers += [tf.matmul(tf.transpose(matrix), matrix)]
        values = session.run(style_gram_layers, feed_dict=model.create_feed_dict(image=style_image))
        style_layer_losses = []
        for value, gram_layer in zip(values, style_gram_layers):
            style_layer_losses.append(tf.reduce_mean(tf.square(gram_layer - tf.constant(value))))
        loss_style = tf.reduce_mean(style_layer_losses)

    # generate de-noise loss
    loss_denoise = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))

    # calculate difference
    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')
    session.run([adj_content.initializer, adj_style.initializer, adj_denoise.initializer])
    loss_combined = weight_content * adj_content * loss_content + weight_style * adj_style * loss_style + weight_denoise * adj_denoise * loss_denoise
    run_list = [tf.gradients(loss_combined, model.input),
                adj_content.assign(1.0 / (loss_content + 1e-10)),  # content adjustments
                adj_style.assign(1.0 / (loss_style + 1e-10)),  # style adjustments
                adj_denoise.assign(1.0 / (loss_denoise + 1e-10))]  # de-noise adjustments

    # apply via gradient
    result = np.random.rand(*content_image.shape) + 128
    for i in range(num_iterations):
        grad, adj_content_val, adj_style_val, adj_denoise_val = session.run(run_list, feed_dict=model.create_feed_dict(image=result))
        grad = np.squeeze(grad)
        result -= grad * (step_size / (np.std(grad) + 1e-8))
        result = np.clip(result, 0.0, 255.0)
    session.close()
    return result
