function toggleSlice(btn,slice)
{
	elem = document.getElementById(slice);
	if(elem.className == 'slice0')
	{
		elem.className = 'slice1';
		btn.className = 'btn1';
	}
	else
	{
		elem.className = 'slice0';
		btn.className = 'btn0';
	}
}

function highlight_same(obj)
{
	if(this.prev != null)
	{
		var names = document.getElementsByName(this.prev);
		for(i = 0 ; i < names.length; i++)
		{
			names[i].style.backgroundColor = 'initial';
		}
	}
	if(obj.getAttribute('name') != null)
	{
		var names = document.getElementsByName(obj.getAttribute('name'));
		for(i = 0 ; i < names.length; i++)
		{
			if(names.length > 1)
				names[i].style.backgroundColor = 'Yellow';
			else
				names[i].style.backgroundColor = 'Red';
		}
		this.prev = obj.getAttribute('name');
	}
}
