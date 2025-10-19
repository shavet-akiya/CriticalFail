import Link from 'next/link'

export default function NotFound() {
    return (
        <div className='select-none'>
            <p className='obsidian-colour text-3xl'>You've hit a dead end!</p>
            <p className='text-gray-400'>Let's get you back home</p>

            <button>
                <Link href="/" className='btn btn-primary'>Return Home</Link>
            </button>
        </div>
    )
}