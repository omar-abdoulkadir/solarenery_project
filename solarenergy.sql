# HeidiSQL Dump 
#
# --------------------------------------------------------
# Host:                 127.0.0.1
# Database:             solarenergy
# Server version:       5.4.3-beta-community
# Server OS:            Win32
# Target-Compatibility: Standard ANSI SQL
# HeidiSQL version:     3.1 RC1 Revision: 1064
# --------------------------------------------------------

/*!40100 SET CHARACTER SET latin1;*/
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ANSI';*/
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;*/


#
# Database structure for database 'solarenergy'
#

CREATE DATABASE /*!32312 IF NOT EXISTS*/ "solarenergy" /*!40100 DEFAULT CHARACTER SET latin1 */;

USE "solarenergy";


#
# Table structure for table 'dataset'
#

CREATE TABLE /*!32312 IF NOT EXISTS*/ "dataset" (
  "s1" varchar(50) DEFAULT NULL,
  "s2" varchar(50) DEFAULT NULL,
  "s3" varchar(50) DEFAULT NULL,
  "s4" varchar(50) DEFAULT NULL,
  "s5" varchar(50) DEFAULT NULL,
  "s6" varchar(50) DEFAULT NULL,
  "s7" varchar(500) DEFAULT NULL,
  "s8" varchar(500) DEFAULT NULL
) /*!40100 DEFAULT CHARSET=latin1*/;



#
# Dumping data for table 'dataset'
#

# (No data found.)



#
# Table structure for table 'resultinfo'
#

CREATE TABLE /*!32312 IF NOT EXISTS*/ "resultinfo" (
  "flimname" varchar(50) DEFAULT NULL,
  "review" varchar(50) DEFAULT NULL,
  "result" varchar(50) DEFAULT NULL
) /*!40100 DEFAULT CHARSET=latin1*/;



#
# Dumping data for table 'resultinfo'
#

# (No data found.)



#
# Table structure for table 'user_information'
#

CREATE TABLE /*!32312 IF NOT EXISTS*/ "user_information" (
  "userid" int(11) NOT NULL AUTO_INCREMENT,
  "username" varchar(50) DEFAULT NULL,
  "password" varchar(50) DEFAULT NULL,
  "address" varchar(50) DEFAULT NULL,
  "emailid" varchar(50) DEFAULT NULL,
  "mobile" varchar(50) DEFAULT NULL,
  PRIMARY KEY ("userid")
) AUTO_INCREMENT=3 /*!40100 DEFAULT CHARSET=latin1*/;



#
# Dumping data for table 'user_information'
#

/*!40000 ALTER TABLE "user_information" DISABLE KEYS;*/
LOCK TABLES "user_information" WRITE;
REPLACE INTO "user_information" ("userid", "username", "password", "address", "emailid", "mobile") VALUES
	(1,'priya','priya','cbe','priya@gmail.com','9856325698');
REPLACE INTO "user_information" ("userid", "username", "password", "address", "emailid", "mobile") VALUES
	(2,'Ranji','ranji','cbe','ranji@gmail.com','9865748514');
UNLOCK TABLES;
/*!40000 ALTER TABLE "user_information" ENABLE KEYS;*/
/*!40101 SET SQL_MODE=@OLD_SQL_MODE;*/
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;*/
