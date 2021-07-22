#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pymysql
# 打开数据库连接
db = pymysql.connect(host="localhost",user="root",password="123177419",database="login" )
# 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()
# 使用 execute() 方法执行 SQL，如果表存在则删除
cursor.execute("DROP TABLE IF EXISTS users")
# 使用预处理语句创建表
sql = """CREATE TABLE users (
         username  CHAR(20) NOT NULL,
         password  CHAR(20) NOT NULL,
          email  CHAR(20)  NOT NULL)"""
cursor.execute(sql)
# 关闭数据库连接
db.close()