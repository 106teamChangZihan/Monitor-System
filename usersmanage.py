import pymysql
from sqltest import *
#修改密码和删除账号

config ={ "host": "localhost",
    "user": "root",
    "password": "123177419",
    "db": "login"}
conn = pymysql.connect(**config)
cur = conn.cursor()

def changepassword(username):
    if(adminfind(username)):
        sql = "UPDATE admin SET password = '%s' WHERE username = '%s'" % (password,username)
        try:
            cur.execute(sql)
            conn.commit()
        except:
            conn.rollback()
        conn.close()
    elif(usersfind(username)):
        sql = "UPDATE users SET password = '%s' WHERE username = '%s'" % (password,username)
        try:
            cur.execute(sql)
            conn.commit()
        except:
            conn.rollback()
        conn.close()
        return True
    else:
        return False



def deleteusers(username):
    if(users(username)):
        sql = "DELETE FROM employee WHERE username = '%s'" % (username)
        try:
            cur.execute(sql)
            conn.commit()
        except:
            conn.rollback()
        conn.close()
        return True
    else:
        return False



def users(username):
    if usersfind(username):
        return  True

def usersfind(username):
	sql="SELECT * FROM users WHERE username ='%s'" %(username)
	cur.execute(sql)
	result = cur.fetchall()
	if (len(result) == 0):
		return False
	else:
		return True


def adminfind(username):
    sql = "SELECT * FROM admin WHERE username ='%s'" % (username)
    cur.execute(sql)
    result = cur.fetchall()
    if (len(result) == 0):
        return False
    else:
        return True