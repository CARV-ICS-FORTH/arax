package Arax;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import java.util.List;
import java.util.Arrays;

public abstract class AraxObject
{
	public class cRep extends Structure
	{
		public Pointer repo;
		public Pointer prev;
		public Pointer next;
		public Pointer owner;
		public long alloc_size;
		/* TODO:Probably should make the above a seperate struct */
		public int type;
		public int ref_count;
		public Pointer name;	// Not sure if 'proper'
		public cRep(Pointer ptr)
		{
			super(ptr);
			read();
		}
		protected List<String> getFieldOrder()
		{
			return Arrays.asList(new String[] {"repo", "prev", "next", "owner", "alloc_size","type","ref_count","name"});
		}
	}

	public AraxObject(Pointer ptr)
	{
		this.ptr = ptr;
		crep = new cRep(ptr);
	}

	public String getName()
	{
		return crep.name.getString(0);
	}

	public Pointer getPointer()
	{
		return ptr;
	}

	public void setPointer(Pointer ptr)
	{
		this.ptr = ptr;
	}

	public String toString()
	{
		return this.getClass().getName()+"("+getName()+","+crep.type+")@0x"+Long.toHexString(Pointer.nativeValue(crep.owner));
	}
	public abstract void release();
	private Pointer ptr;
	private cRep crep;
}
