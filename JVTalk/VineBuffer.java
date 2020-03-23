package Vinetalk;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import java.util.List;
import java.util.Arrays;
import com.sun.jna.Memory;
import com.sun.jna.Native;

public class VineBuffer
{
	public Pointer user_buffer;
	public long user_buffer_size;
	public Pointer vine_data;
	private Object juser_buffer;
	private Class juser_class;

	private void init(Object jo, Pointer user,long size,Class type)
	{
		user_buffer = user;
		user_buffer_size = size;
		vine_data = VineTalkInterface.INSTANCE.VINE_BUFFER(user,size);
		juser_buffer = jo;
		juser_class = type;
	}

	public VineBuffer(Structure struct)
	{
		this(struct,1);	// Synchronize by default
	}

	public VineBuffer(Structure struct, int elementNo)
	{
		init(
			struct,
			struct.getPointer(),
			struct.size()*elementNo,
			Structure.class
		);
	}

	public VineBuffer(byte [] data)
	{
		init(
			data,
			new Memory(data.length),
			data.length,
			Byte.class
		);
		user_buffer.write(0,data,0,data.length);
	}

	public VineBuffer(float [] data)
	{
		int bytes = data.length*Native.getNativeSize(Float.class);
		init(
			data,
			new Memory(bytes),
			bytes,
			Float.class
		);
		user_buffer.write(0,data,0,data.length);
	}

	public VineBuffer(long [] data)
	{
		int bytes = data.length*Native.getNativeSize(Long.class);
		init(
			data,
			new Memory(bytes),
			bytes,
			Long.class
		);
		user_buffer.write(0,data,0,data.length);
	}

	public VineBuffer(double [] data)
	{
		int bytes = data.length*Native.getNativeSize(Double.class);
		init(
			data,
			new Memory(bytes),
			bytes,
			Double.class
		);
		user_buffer.write(0,data,0,data.length);
	}

	public VineBuffer(int [] data)
	{
		int bytes = data.length*Native.getNativeSize(Integer.class);
		init(
			data,
			new Memory(bytes),
			bytes,
			Integer.class
		);
		user_buffer.write(0,data,0,data.length);
	}

	public void cloneFrom(VineBuffer source)
	{
		user_buffer = source.user_buffer;
		user_buffer_size = source.user_buffer_size;
		vine_data = source.vine_data;
		juser_buffer = source.juser_buffer;
		juser_class = source.juser_class;
	}

	private void read()
	{
		if(user_buffer == null)
			return;

		if(juser_class == Byte.class)
		{
			byte [] data = user_buffer.getByteArray(0,(int)user_buffer_size);
			if(juser_buffer != null)
				System.arraycopy(data,0,juser_buffer,0,(int)user_buffer_size);
		}
		else
		if(juser_class == Float.class)
		{
			int elements = (int)user_buffer_size/Native.getNativeSize(Float.class);
			float [] data = user_buffer.getFloatArray(0,elements);
			if(juser_buffer != null) {
				System.arraycopy(data,0,juser_buffer,0,elements);
			}
		}
		else
		if(juser_class == Long.class)
		{
			int elements = (int)user_buffer_size/Native.getNativeSize(Long.class);
			long [] data = user_buffer.getLongArray(0,elements);
			if(juser_buffer != null)
				System.arraycopy(data,0,juser_buffer,0,elements);
		}
		else
		if(juser_class == Double.class)
		{
			int elements = (int)user_buffer_size/Native.getNativeSize(Double.class);
			double [] data = user_buffer.getDoubleArray(0,elements);
			if(juser_buffer != null)
				System.arraycopy(data,0,juser_buffer,0,elements);
		}
		else
		if(juser_class == Integer.class)
		{
			int elements = (int)user_buffer_size/Native.getNativeSize(Integer.class);
			int [] data = user_buffer.getIntArray(0,elements);
			if(juser_buffer != null)
				System.arraycopy(data,0,juser_buffer,0,elements);
		}
		else
		if(juser_class == Structure.class)
		{
			if(juser_buffer != null) {
			    Structure data = (Structure) juser_buffer;
			    data.read();
		   }
	    } else {
            assert null!="Invalid juser_class!";
	    }
	}

	public enum VineDataFlags {
		NONE_SYNC(0),
		USER_SYNC(1),
		SHM_SYNC(2),
		REMT_SYNC(4),
		ALL_SYNC(7),
		FREE(8);

		private final int value;

		VineDataFlags(int value)
		{this.value = value;}
		int getAsInt()
		{return value;}
	}

	public void modified(VineDataFlags where)
	{
		VineTalkInterface.INSTANCE.vine_data_modified(vine_data,where.getAsInt());
	}

	public void syncToRemote(VineAccelerator accel,boolean wait)
	{
		VineTalkInterface.INSTANCE.vine_data_sync_to_remote(accel.getPointer(),vine_data,(wait)?1:0);
	}

	public void syncFromRemote(VineAccelerator accel,boolean wait)
	{
		VineTalkInterface.INSTANCE.vine_data_sync_from_remote(accel.getPointer(),vine_data,(wait)?1:0);
		read();
	}

	public Pointer getPointer()
	{
		return vine_data;
	}
}
