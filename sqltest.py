#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pymysql
# 连接数据库
config ={ "host": "localhost",
    "user": "root",
    "password": "123177419",
    "db": "login"}

conn = pymysql.connect(**config)
cur = conn.cursor()


def add_user(username, password, email):
    # sql commands
    sql = "INSERT INTO users(username, password, email)  \
           VALUES ('%s','%s','%s')" %(username, password, email)
    # execute(sql)
    cur.execute(sql)
    # commit
    conn.commit()  # 对数据库内容有改变，需要commit()
    conn.close()


def add_admin(username, password, email):
    # sql commands
    sql = "INSERT INTO admin(username, password, email)  \
           VALUES ('%s','%s','%s')" %(username, password, email)
    # execute(sql)
    cur.execute(sql)
    # commit
    conn.commit()  # 对数据库内容有改变，需要commit()
    conn.close()

#选择身份
def isusers():
    #选中用户则为true
    return True

def isadmin():
    #选中管理员则为true
    return True

def is_existed_users(username,password):
	sql="SELECT * FROM users WHERE username ='%s'  \
         and password ='%s'" %(username,password)
	cur.execute(sql)
	result = cur.fetchall()
	if (len(result) == 0):
		return False
	else:
		return True

def is_existed_admin(username,password):
    sql = "SELECT * FROM admin WHERE username ='%s'  \
           and password ='%s'" % (username, password)
    cur.execute(sql)
    result = cur.fetchall()
    if (len(result) == 0):
        return False
    else:
        return True
