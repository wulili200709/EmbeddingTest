
#ifndef NKIOLIB_H
#define NKIOLIB_H

#if defined _WIN32

#include <Windows.h>
#ifdef NKIOLIB_EXPORTS
#define NK_API __declspec(dllexport)
#else
#define NK_API __declspec(dllimport)
#endif

#elif defined __linux__
#define NK_API
#else
#define NK_API
#endif

// Error Codes
#define NKIO_ENOERR				0		/*!< No error. */
#define NKIO_EBUSY				1		/*!< Bus is Busy. */
#define NKIO_EINUSED			2		/*!< Bus is in used. */
#define NKIO_EIO				3		/*!< Bus is error. */
#define NKIO_ETIMEOUT			4		/*!< Timeout error occurred. */
#define NKIO_EDEVERR			5		/*!< IO Device is not accessed. */
#define NKIO_EINVAL				6		/*!< Illegal argument. */
#define NKIO_EFILE				7		/*!< File is read error. */


#ifdef __cplusplus
extern "C" {
#endif

	/**
	* @brief: NKIO library and driver initialized by the configure file.
	* @param: configFile - The path of the IO board configure file.
	*
	* @return: return the NKIO_ENOERR if succeed, otherwise, return the errorcode, please see the error codes.
	*
	* @note: Must be called once firstly before calling the read and write DIO functions.
	*/
	NK_API int NKDIO_LibraryInit(const char *configFile);
	/**
	* @brief: deinitialize the driver and release the resources allocated in the NKIO library.
	*
	*/
	NK_API void NKDIO_LibraryDeinit(void);

	/************************************************************************************************************/
	/*           General IO                                                                                     */
	/************************************************************************************************************/
	/**
	* @brief: Read the DI port in the polling mode.
	* @param: diByteIndex - the index of the port, there are 8 channels in each port, the value is started from 0.
	*		  pByteValue - the address to store the read byte value.
	*
	* @return: return the NKIO_ENOERR if succeed, otherwise, return the errorcode, please see the error codes.
	*
	*/
	NK_API int NKDIO_PollingReadDiByte(unsigned char diByteIndex,unsigned char *pByteValue);
	
	/**
	* @brief: Read the 2 ports value in the polling mode.
	* @param: diWordIndex - the index of the word, the value is started from 0.
	*		  pWordValue - the address to store the read word value.
	*
	* @return: return the NKIO_ENOERR if succeed, otherwise, return the errorcode, please see the error codes.
	*
	*/
	NK_API int NKDIO_PollingReadDiWord(unsigned char diWordIndex, unsigned short *pWordValue);

	/**
	* @brief: Write the DO port value in the polling mode.
	* @param: doByteIndex - the index of the port, the value is started from 0.
	*		  doByteValue - the byte value will be write out.
	*
	* @return: return the NKIO_ENOERR if succeed, otherwise, return the errorcode, please see the error codes.
	*
	*/
	NK_API int NKDIO_PollingWriteDoByte(unsigned char doByteIndex, unsigned char doByteValue);

	/**
	* @brief: Write the DO word value in the polling mode.
	* @param: doWordIndex - the index of the word, the value is started from 0.
	*		  doWordValue - the word value will be write out.
	*
	* @return: return the NKIO_ENOERR if succeed, otherwise, return the errorcode, please see the error codes.
	*
	*/
	NK_API int NKDIO_PollingWriteDoWord(unsigned char doWordIndex, unsigned short doWordValue);

	/**
	* @brief: Read back the DO port value in the polling mode.
	* @param: doByteIndex - the index of the port, the value is started from 0.
	*		  pByteValue - the address to store the read value.
	*
	* @return: return the NKIO_ENOERR if succeed, otherwise, return the errorcode, please see the error codes.
	*
	*/
	NK_API 	int NKDIO_PollingReadDoByte(unsigned char doByteIndex, unsigned char *pByteValue);

	/**
	* @brief: Read back the DO word value in the polling mode.
	* @param: doWordIndex - the index of the word, the value is started from 0.
	*		  pWordValue - the address to store the read value.
	*
	* @return: return the NKIO_ENOERR if succeed, otherwise, return the errorcode, please see the error codes.
	*
	*/
	NK_API 	int NKDIO_PollingReadDoWord(unsigned char doWordIndex, unsigned short *pWordValue);



#ifdef __cplusplus
}
#endif

#endif // NKIOLIB_H

