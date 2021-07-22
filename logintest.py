from flask import Flask,render_template,Response
from flask import redirect
from flask import url_for
from flask import request

from sqltest import *
from camera import *
#登录、注册功能
#
# create Flask instance
#
app = Flask(__name__)

#
# route path
#

@app.route('/')
def index():
    return redirect( url_for('user_login') )



@app.route('/user_login',methods=['GET','POST'])
def user_login():
    if request.method=='POST':  # 注册发送的请求为POST请求
        # get the user information that user submitted
        username = request.form['username']
        password = request.form['password']
        # if the information is correct(in our database)
        if isusers():     #用户
            if is_existed_users(username, password):
               return render_template('video.html')  #跳转到主页-用户权限
            else:
                login_massage = "Login Failed"
                return render_template('index.html', message=login_massage)  #重新登录
        elif isadmin():   #管理员
            if is_existed_admin(username,password):
                return render_template('video.html')
            else:
                login_massage = "Login Failed"
                return render_template('index.html', message=login_massage)
        else:
            login_massage = "Login Failed"
            return render_template('index.html', message=login_massage)
    return render_template('index.html')


@app.route("/regiser",methods=["GET", 'POST'])
def register():
    if(isusers()):
        if(userfind()):
            return False
        else:
            if request.method == 'POST':
                if len(username)>=20:  #用户名过长
                    return False
                else:
                    # 上面先执行邮箱认证再执行下一条
                    add_user(request.form['username'], request.form['password'], request.form['email'])
                    return render_template('registersuccessfully.html')
    elif(isadmin()):
        if(adminfind()):
            return False
        else:
            if request.method == 'POST':
                if len(username)>=20:
                    return False
                else:
                    # 上面先执行邮箱认证再执行下一条
                    add_admin(request.form['username'], request.form['password'], request.form['email'])
                    return render_template('registersuccessfully.html')

    return render_template('register.html')



@app.route('/video_start')
def video_start():
    # 通过将一帧帧的图像返回，就达到了看视频的目的。
    # multipart/x-mixed-replace是单次的http请求-响应模式，如果网络中断，会导致视频流异常终止，必须重新连接才能恢复
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(debug=True)
